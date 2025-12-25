import torch
import torch.nn as nn
import torch.nn.functional as F 

from .autoencoder import RoPEMultiheadAttention, RoPETransformerEncoderLayer
from .autoencoder import Unet_Enc, Unet_Dec, ResidualBlock
from .model_utils import init_weights

from .DynNet import DynNet, UPredNet

from .CFB import FusionBlockBottleneck, ChannelFusionBlock

## FULL SPATIOTEMPORAL MODEL
## The SWAU-Net architecture is designed for video prediction in a low-data, short-clip regime (I0-I3).
## Core Concept: Use Sliding Window Attention (SWA) to regularize/augment features (F) into integrated 
## features (M), which are then evolved (E) to predict future frames. The model enforces causality 
## through the structure of the SWA input and performs simultaneous reconstruction (I0_hat).

# ------------------------------------------------------------------------------
# SLIDING WINDOW ATTENTION    
# ------------------------------------------------------------------------------

class LocalSpatioTemporalMixer(nn.Module):
    # Changed in_channels default to None
    def __init__(self, in_channels=None, kernel_size=3):
        super().__init__()
        
        C = in_channels if in_channels is not None else 64 # Assuming a common default for L3
        self.in_channels = C
        
        self.local_proj = nn.Conv2d(C * 3, C * 3, 
                                    kernel_size=kernel_size, padding=kernel_size//2, 
                                    groups=C * 3, bias=False)
        self.pointwise_proj = nn.Conv2d(C * 3, C * 3, kernel_size=1)
        
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(C * 3, C, kernel_size=1),
            nn.GELU(),
            ResidualBlock(C, C)
        )
        self.apply(init_weights)

    def forward(self, F_t_minus_2, F_t_minus_1, F_t):
        F_stacked = torch.cat([F_t_minus_2, F_t_minus_1, F_t], dim=1)
        F_local = self.pointwise_proj(self.local_proj(F_stacked))
        M_t = self.temporal_fusion(F_local)
        M_t = M_t + F_t 
        return M_t
    
# ---------------------------------------------------------------------------    
# ---------------------------------------------------------------------------    
    
class SpatioTemporalGatedMixer(nn.Module):
    """
    Introduces local spatial inductive bias (convolution) within the Transformer block.
    Uses a LEARNABLE SCALAR GATE (mixer_weight) initialized small to control
    the fusion, prioritizing the attention output initially.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        
        # Core convolutional path
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1), # Pointwise projection
            nn.BatchNorm2d(channels)
        )
        
        # Learnable Scalar Gate for Adaptive Fusion - Initialized near 0
#         self.mixer_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.mixer_weight = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, attn_output_tokens, D1, D2):
        """
        Args:
            attn_output_tokens (Tensor): The raw attention output, shape (B*Axial_Size, L, C).
            D1 (int): Dimension 1 (e.g., Time, T).
            D2 (int): Dimension 2 (e.g., Spatial, W or H).
        """
        B_A, L, C = attn_output_tokens.shape
        T_dim, S_dim = D1, D2

        # 1. Reshape to 4D CNN format (B*Axial_Size, C, D1, D2)
        x_4d_reshaped = attn_output_tokens.reshape(B_A, T_dim, S_dim, C).permute(0, 3, 1, 2)
        
        # 2. Convolutional Mixing
        x_conv = self.conv(x_4d_reshaped)
        
        # 3. Reshape back to tokens (B*Axial_Size, L, C)
        x_out_tokens = x_conv.permute(0, 2, 3, 1).flatten(1, 2)
        
        # 4. Apply Gated Fusion: Fused = Attention_Output + (Learned_Weight * Convolutional_Output)
        # This output serves as a refined residual update to the input feature.
        fused_output = attn_output_tokens + (self.mixer_weight * x_out_tokens)
        
        return fused_output

class AxialTemporalSWAInterleavedLayer(nn.Module):
    """
    Adjusted SWA layer to prioritize attention by correcting the residual path 
    around the attention and mixer blocks.
    """
    def __init__(self, in_channels, nhead, dim_feedforward):
        super().__init__()
        # SWA Attention Blocks
        self.attn_tw = RoPEMultiheadAttention(in_channels, nhead, dropout=0, batch_first=True)
        self.norm_tw = nn.LayerNorm(in_channels)
        self.attn_th = RoPEMultiheadAttention(in_channels, nhead, dropout=0, batch_first=True)
        self.norm_th = nn.LayerNorm(in_channels)
        
        # Mixer blocks after each Attention pass (Uses the gated mixer)
        self.mixer_tw = SpatioTemporalGatedMixer(in_channels)
        self.mixer_th = SpatioTemporalGatedMixer(in_channels)

        self.ffn = nn.Sequential(
            nn.Linear(in_channels, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, in_channels),
            nn.Dropout(0)
        )
        self.norm_ffn = nn.LayerNorm(in_channels)

    def forward(self, input_tokens, B, T, H, W):
        C = input_tokens.shape[-1]
        residual_input = input_tokens.reshape(B, T * H * W, C) # (B, L, C)

        # 1. Time-Width Attention (TW) - Axial on H
        tokens_tw = residual_input.reshape(B, H, W, T, C).permute(0, 1, 3, 2, 4).reshape(B * H, T * W, C)
        attn_out_tw, _ = self.attn_tw(tokens_tw, tokens_tw, tokens_tw)
        
        # --- MIXING STEP 1 (Attention + Gated Conv Delta) ---
        # The mixer calculates (attn_out + gated_conv_delta)
        mixer_out_tw = self.mixer_tw(attn_out_tw, T, W)  
        
        # Final normalization: (Input) + (Attention + Gated_Conv_Delta)
        # This is a robust residual path that includes the full attention result.
        attn_out_tw_norm = self.norm_tw(tokens_tw + mixer_out_tw)
        
        interim_tokens = attn_out_tw_norm.reshape(B, H, T, W, C).permute(0, 2, 1, 3, 4)

        # 2. Time-Height Attention (TH) - Axial on W
        tokens_th = interim_tokens.permute(0, 3, 1, 2, 4).reshape(B * W, T * H, C)
        attn_out_th, _ = self.attn_th(tokens_th, tokens_th, tokens_th)
        
        # --- MIXING STEP 2 ---
        mixer_out_th = self.mixer_th(attn_out_th, T, H)
        
        # Final normalization
        attn_out_th_norm = self.norm_th(tokens_th + mixer_out_th)
        
        final_tokens = attn_out_th_norm.reshape(B, W, T, H, C).permute(0, 2, 3, 1, 4)  
        
        # 3. FFN
        final_tokens_flat = final_tokens.reshape(B, T * H * W, C)
        ffn_out = self.ffn(final_tokens_flat)
        
        # Residual 3 (FFN)
        output_tokens = self.norm_ffn(final_tokens_flat + ffn_out)
        return output_tokens.reshape(B, T, H, W, C)
    
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class InterleavedAxialTemporalSWAIntegrator(nn.Module):
    def __init__(self, in_channels, nhead, dim_feedforward, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            AxialTemporalSWAInterleavedLayer(in_channels, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        
    def forward(self, F_t_minus_2, F_t_minus_1, F_t):
        # F_t_minus_2, F_t_minus_1, F_t are the raw encoder features (F).
        # F_t is the raw feature for the current time step (the third input).
        stacked_features = torch.stack([F_t_minus_2, F_t_minus_1, F_t], dim=1)
        B, T, C, H, W = stacked_features.shape
        tokens = stacked_features.permute(0, 1, 3, 4, 2)

        for layer in self.layers:
            tokens = layer(tokens, B, T, H, W)  # Temporal mixing
            
        # M_t is the aggregated state M(t) derived from the last time step tokens.
        M_t_tokens = tokens[:, 2, :, :, :]
        M_t = M_t_tokens.permute(0, 3, 1, 2)
        
        # --- MACRO-RESIDUAL ADDITION ---
        # M_t = M_t + F_t: Recovers high-frequency spatial detail by blending 
        # the stable aggregated state (M_t) with the raw encoder input (F_t).
        M_t = M_t + F_t

        return M_t

class FullGlobalSWAIntegrator(nn.Module):
    """
    Implements Full Global Spatio-Temporal Attention over a 3-frame sequence 
    (T*H*W tokens). This is used only for the L5 Bottleneck (e.g., 16x16x3 frames).
    """
    def __init__(self, in_channels, nhead, dim_feedforward):
        super().__init__()
        # Uses the existing single-layer transformer encoder for full self-attention
        self.attn_layer = RoPETransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0,
            batch_first=True
        )
        # Learnable scalar for Macro-Residual (M_t = M_t + w * F_t)
        # Initialized near 0.5 for balance, as it's a critical macro-skip.
        self.macro_residual_weight = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, F_t_minus_2, F_t_minus_1, F_t):
        # Stack 3 frames (F_t-2, F_t-1, F_t): [B, 3, C, H, W]
        stacked_features = torch.stack([F_t_minus_2, F_t_minus_1, F_t], dim=1)
        B, T, C, H, W = stacked_features.shape
        
        # 1. Flatten into a single token sequence: [B, T*H*W, C] 
        input_tokens = stacked_features.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)

        # 2. Perform Full Global Spatio-Temporal Attention
        output_tokens = self.attn_layer(input_tokens)

        # 3. Extract the aggregated state M_t (last time step, index 2)
        # Reshape back to [B, T, H, W, C]
        output_tokens_reshaped = output_tokens.reshape(B, T, H, W, C)
        
        # M_t is the aggregated token for the current frame F_t (index 2)
        M_t_tokens = output_tokens_reshaped[:, 2, :, :, :] 
        M_t = M_t_tokens.permute(0, 3, 1, 2) # [B, C, H, W]

        # 4. Macro-Residual (M_t = M_t + w * F_t) - blending aggregated state with raw input
        M_t = M_t + (self.macro_residual_weight * F_t)

        return M_t

# ---------------------------------------------------------------------------

class SlidingWindowAttention(nn.Module):
    # Changed base_channels default to None
    def __init__(self, args, base_channels=None):
        super().__init__()
        C = base_channels if base_channels is not None else 16 # Assuming default is 16
        NUM_ATTN_LAYERS = 2
        
        self.attn_bn = InterleavedAxialTemporalSWAIntegrator(
            in_channels=C * 16, 
            nhead=args.nhead, 
            dim_feedforward=args.d_attn2, 
            num_layers=NUM_ATTN_LAYERS
        )
        
        self.attn_l4 = InterleavedAxialTemporalSWAIntegrator(
            in_channels=C * 8, 
            nhead=args.nhead, 
            dim_feedforward=args.d_attn2, 
            num_layers=NUM_ATTN_LAYERS
        )
        
        self.mixer_l3 = LocalSpatioTemporalMixer(in_channels=C * 4, kernel_size=3) 
        
        # --- L2 Output: Set to C=16 channels to match DynNet/Recurrence. ---
        self.conv_l2 = nn.Sequential(
            nn.Conv2d(C * 3, C * 1, kernel_size=1), # Input 48 (3C) -> Output 16 (C*1)
            nn.GELU(),
            ResidualBlock(C * 1, C * 1) # Final output is 16 channels.
        )
        
        # --- L1 Output: Set to C/2=8 channels to match DynNet/Recurrence. ---
        self.conv_l1 = nn.Sequential(
            nn.Conv2d(C * 3 // 2, C // 2, kernel_size=1), # Input 24 (1.5C) -> Output 8 (C/2)
            nn.GELU(),
            ResidualBlock(C // 2, C // 2) # Final output is 8 channels.
        )
        
        self.apply(init_weights) 

    def forward(self, F_t_minus_2, F_t_minus_1, F_t):
        f1_t_2, f2_t_2, f3_t_2, f4_t_2, bn_t_2 = F_t_minus_2
        f1_t_1, f2_t_1, f3_t_1, f4_t_1, bn_t_1 = F_t_minus_1
        f1_t, f2_t, f3_t, f4_t, bn_t = F_t
        
        m_bn = self.attn_bn(bn_t_2, bn_t_1, bn_t)
        m4 = self.attn_l4(f4_t_2, f4_t_1, f4_t)
        m3 = self.mixer_l3(f3_t_2, f3_t_1, f3_t) 

        f2_cat = torch.cat([f2_t_2, f2_t_1, f2_t], dim=1)
        m2 = self.conv_l2(f2_cat) # Output is now C=16 channels

        f1_cat = torch.cat([f1_t_2, f1_t_1, f1_t], dim=1)
        m1 = self.conv_l1(f1_cat) # Output is now C/2=8 channels
        
        return m1, m2, m3, m4, m_bn

# class SlidingWindowAttention(nn.Module):
#     """
#     The temporal core of SWAU-Net. Aggregates features M_t from the 3-frame 
#     window. Uses Full Global Attention at L5 and Axial Attention at L4.
#     """
#     def __init__(self, args, base_channels=None):
#         super().__init__()
#         C = base_channels if base_channels is not None else 16
#         NUM_ATTN_LAYERS = 2
        
#         # --- L5 Bottleneck Attention (FULL GLOBAL SPATIO-TEMPORAL) ---
#         # Replacing InterleavedAxialTemporalSWAIntegrator with the new class
#         self.attn_bn = FullGlobalSWAIntegrator(
#             in_channels=C * 16, # L5 Channel Count (e.g., 256)
#             nhead=args.nhead, 
#             dim_feedforward=args.d_attn2, 
#         )
        
#         # --- L4 Attention (RETAINS AXIAL for computational efficiency) ---
#         # Note: InterleavedAxialTemporalSWAIntegrator is assumed to be defined elsewhere.
#         self.attn_l4 = InterleavedAxialTemporalSWAIntegrator(
#             in_channels=C * 8, # L4 Channel Count (e.g., 128)
#             nhead=args.nhead, 
#             dim_feedforward=args.d_attn2, 
#             num_layers=NUM_ATTN_LAYERS
#         )
        
#         # --- L3 Mixer (Retains local convolutional mixing) ---
#         # Note: LocalSpatioTemporalMixer is assumed to be defined elsewhere.
#         self.mixer_l3 = LocalSpatioTemporalMixer(in_channels=C * 4, kernel_size=3)
        
#         # --- L2 & L1 Convs (Retains simple convolutional aggregation) ---
#         self.conv_l2 = nn.Sequential(
#             nn.Conv2d(C * 3, C * 1, kernel_size=1),
#             nn.GELU(),
#             ResidualBlock(C * 1, C * 1)
#         )
        
#         self.conv_l1 = nn.Sequential(
#             nn.Conv2d(C * 3 // 2, C // 2, kernel_size=1),
#             nn.GELU(),
#             ResidualBlock(C // 2, C // 2)
#         )
        
#         self.apply(init_weights)

#     def forward(self, F_t_minus_2, F_t_minus_1, F_t):
#         f1_t_2, f2_t_2, f3_t_2, f4_t_2, bn_t_2 = F_t_minus_2
#         f1_t_1, f2_t_1, f3_t_1, f4_t_1, bn_t_1 = F_t_minus_1
#         f1_t, f2_t, f3_t, f4_t, bn_t = F_t
        
#         # L5: Full Global Spatio-Temporal Attention
#         m_bn = self.attn_bn(bn_t_2, bn_t_1, bn_t)
        
#         # L4: Axial Spatio-Temporal Attention
#         m4 = self.attn_l4(f4_t_2, f4_t_1, f4_t)
        
#         # L3: Convolutional Mixer
#         m3 = self.mixer_l3(f3_t_2, f3_t_1, f3_t)
        
#         # L2: Concatenation and Convolution
#         f2_cat = torch.cat([f2_t_2, f2_t_1, f2_t], dim=1)
#         m2 = self.conv_l2(f2_cat)
        
#         # L1: Concatenation and Convolution
#         f1_cat = torch.cat([f1_t_2, f1_t_1, f1_t], dim=1)
#         m1 = self.conv_l1(f1_cat)
        
#         return m1, m2, m3, m4, m_bn
    
# ---------------------------------------------------------------------------

class SWAU_Net(nn.Module):
    """
    Sliding Window Attention U-Net (SWAU_Net)
    The full Video Prediction system with Dual Channel Fusion Stages (CFB_enc, CFB_dec).
    The forward pass is set to ensure the targets tensor is structurally aligned 
    (N, C, T, H, W) with the predictions.
    """
    # Changed img_channels and base_channels defaults to None
    def __init__(self, args, img_channels=None, base_channels=None):
        super().__init__()
        
        # Resolve defaults inside __init__
        if img_channels is None:
            img_channels = 3 # Hardcoding initial default value
        if base_channels is None:
            base_channels = 16 # Hardcoding initial default value
            
        # E1: Feature Extractor
        self.E1 = Unet_Enc(args, img_channels, base_channels)
        
        # CFB_enc: Pre-Dynamics Fusion
        self.CFB_enc = ChannelFusionBlock(base_channels) 
        
        # CFB_dec: Post-Dynamics Refinement
        self.CFB_dec = ChannelFusionBlock(base_channels) 
        
        # SWA: Feature Aggregator (Temporal Core)
        self.SWA = SlidingWindowAttention(args, base_channels)
        # P: Temporal Feature Predictor
        self.P = DynNet(args, base_channels)
        # D1: Frame Reconstructor
        self.D1 = Unet_Dec(args, img_channels, base_channels)
        
    def forward(self, input_clips):
        B, C, T, H, W = input_clips.shape
        T_pred = T - 1 # 3 predicted frames (I1, I2, I3)
        
        # --- A. FEATURE EXTRACTION & INPUT PREP ---
        I0_gt = input_clips[:, :, 0, :, :] 
        
        # Input to E1: Permute to (N, T, C, H, W) then reshape to (N*T, C, H, W)
        input_frames_E1 = input_clips[:, :, :T_pred, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
        E1_features_flat = self.E1(input_frames_E1)
        
        # --- 1. PRE-DYNAMICS CHANNEL FUSION (CFB_enc) ---
        FUSED_features_flat = self.CFB_enc(E1_features_flat)
        
        # Unpack FUSED features by time step 
        F0_fused = [f[:B] for f in FUSED_features_flat] 
        F1_fused = [f[B:2*B] for f in FUSED_features_flat] 
        F2_fused = [f[2*B:3*B] for f in FUSED_features_flat] 
        
        # --- B. ANCHOR RECONSTRUCTION (I_hat_0) ---
        I0_hat = self.D1(*F0_fused)

        # --- C. SLIDING WINDOW ATTENTION (SWA) ---
        # M0 (Predicts I1): Context is (F0, F0, F0)
        M0_features = self.SWA(F0_fused, F0_fused, F0_fused) 
        
        # M1 (Predicts I2): Context is (F0, F0, F1)
        M1_features = self.SWA(F0_fused, F0_fused, F1_fused) 
        
        # M2 (Predicts I3): Context is (F0, F1, F2)
        M2_features = self.SWA(F0_fused, F1_fused, F2_fused) 

        M_flat = [
            torch.cat([M0_features[i], M1_features[i], M2_features[i]], dim=0) 
            for i in range(5)
        ]
        
        # --- D. TEMPORAL EVOLUTION (DynNet) ---
        Evolved_flat = self.P(*M_flat)
        
        # --- 2. POST-DYNAMICS CHANNEL FUSION (CFB_dec) ---
        Evolved_polished = self.CFB_dec(Evolved_flat)
        
        E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = Evolved_polished

        # --- E. DECODING & RESHAPING ---
        out_frames_pred = self.D1(*Evolved_polished)
        
        # Predictions are built as (N, T, C, H, W) then permuted to (N, C, T, H, W)
        predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        
        # --- TARGET ALIGNMENT FIX ---
        # Targets are sliced to (N, C, T_pred, H, W). No permutation is needed!
        targets = input_clips[:, :, 1:, :, :] 
        
        # Return 9 items
        return predictions, targets, I0_hat, I0_gt, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved
    
    
# ------------------------------------------------------------------------------
#  
# ------------------------------------------------------------------------------


class SWAU_CFB_Ablation(nn.Module):
    """
    Sliding Window Attention U-Net (SWAU_Net)
    The full Video Prediction system with Dual Channel Fusion Stages (CFB_enc, CFB_dec).
    The forward pass is set to ensure the targets tensor is structurally aligned 
    (N, C, T, H, W) with the predictions.
    """
    # Changed img_channels and base_channels defaults to None
    def __init__(self, args, img_channels=None, base_channels=None):
        super().__init__()
        
        # Resolve defaults inside __init__
        if img_channels is None:
            img_channels = 3 # Hardcoding initial default value
        if base_channels is None:
            base_channels = 16 # Hardcoding initial default value
            
        # E1: Feature Extractor
        self.E1 = Unet_Enc(args, img_channels, base_channels)
        
        # SWA: Feature Aggregator (Temporal Core)
        self.SWA = SlidingWindowAttention(args, base_channels)
        # P: Temporal Feature Predictor
        self.P = DynNet(args, base_channels)
        # D1: Frame Reconstructor
        self.D1 = Unet_Dec(args, img_channels, base_channels)
        
    def forward(self, input_clips):
        B, C, T, H, W = input_clips.shape
        T_pred = T - 1 # 3 predicted frames (I1, I2, I3)
        
        # --- A. FEATURE EXTRACTION & INPUT PREP ---
        I0_gt = input_clips[:, :, 0, :, :] 
        
        # Input to E1: Permute to (N, T, C, H, W) then reshape to (N*T, C, H, W)
        input_frames_E1 = input_clips[:, :, :T_pred, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
        FUSED_features_flat = self.E1(input_frames_E1)
        
        # Unpack FUSED features by time step 
        F0_fused = [f[:B] for f in FUSED_features_flat] 
        F1_fused = [f[B:2*B] for f in FUSED_features_flat] 
        F2_fused = [f[2*B:3*B] for f in FUSED_features_flat] 
        
        # --- B. ANCHOR RECONSTRUCTION (I_hat_0) ---
        I0_hat = self.D1(*F0_fused)

        # --- C. SLIDING WINDOW ATTENTION (SWA) ---
        # M0 (Predicts I1): Context is (F0, F0, F0)
        M0_features = self.SWA(F0_fused, F0_fused, F0_fused) 
        
        # M1 (Predicts I2): Context is (F0, F0, F1)
        M1_features = self.SWA(F0_fused, F0_fused, F1_fused) 
        
        # M2 (Predicts I3): Context is (F0, F1, F2)
        M2_features = self.SWA(F0_fused, F1_fused, F2_fused) 

        M_flat = [
            torch.cat([M0_features[i], M1_features[i], M2_features[i]], dim=0) 
            for i in range(5)
        ]
        
        # --- D. TEMPORAL EVOLUTION (DynNet) ---
        Evolved_flat = self.P(*M_flat)
        
        E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = Evolved_flat

        # --- E. DECODING & RESHAPING ---
        out_frames_pred = self.D1(*Evolved_flat)
        
        # Predictions are built as (N, T, C, H, W) then permuted to (N, C, T, H, W)
        predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        
        # --- TARGET ALIGNMENT FIX ---
        # Targets are sliced to (N, C, T_pred, H, W). No permutation is needed!
        targets = input_clips[:, :, 1:, :, :] 
        
        # Return 9 items
        return predictions, targets, I0_hat, I0_gt, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved
    
# ------------------------------------------------------------------------------

    
class SWAU_DynNet_Ablation(nn.Module):
    """
    SWAU-Net DynNet Ablation: Removes the explicit Dynamics Network (DynNet).
    The model is forced to predict the next frame directly from the SWA-aggregated 
    feature state (M_t), thus testing the core inductive bias of decoupling 
    state estimation from temporal evolution.
    """
    def __init__(self, args, img_channels=None, base_channels=None):
        super().__init__()
        
        if img_channels is None:
            img_channels = 3
        if base_channels is None:
            base_channels = 16
            
        # Core Components (reused)
        self.E1 = Unet_Enc(args, img_channels, base_channels)
        self.CFB_enc = ChannelFusionBlock(base_channels) 
        self.CFB_dec = ChannelFusionBlock(base_channels) 
        self.SWA = SlidingWindowAttention(args, base_channels)
        # self.P = DynNet(...) <-- Removed
        self.D1 = Unet_Dec(args, img_channels, base_channels)
        
    def forward(self, input_clips):
        B, C, T, H, W = input_clips.shape
        T_pred = T - 1 # 3 predicted frames (I1, I2, I3)
        
        # --- A. FEATURE EXTRACTION & REFINE (Same as SWAU_Net) ---
        I0_gt = input_clips[:, :, 0, :, :]
        input_frames_E1 = input_clips[:, :, :T_pred, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
        E1_features_flat = self.E1(input_frames_E1)
        FUSED_features_flat = self.CFB_enc(E1_features_flat)
        
        F0_fused = [f[:B] for f in FUSED_features_flat]
        F1_fused = [f[B:2*B] for f in FUSED_features_flat]
        F2_fused = [f[2*B:3*B] for f in FUSED_features_flat]
        
        # --- B. ANCHOR RECONSTRUCTION ---
        I0_hat = self.D1(*F0_fused)

        # --- C. SLIDING WINDOW ATTENTION (SWA) ---
        # Generate the three single-frame aggregated states M0, M1, M2
        M0_features = self.SWA(F0_fused, F0_fused, F0_fused)
        M1_features = self.SWA(F0_fused, F0_fused, F1_fused)
        M2_features = self.SWA(F0_fused, F1_fused, F2_fused)

        M_flat = [
            torch.cat([M0_features[i], M1_features[i], M2_features[i]], dim=0)
            for i in range(5)
        ]
        
        # --- D. TEMPORAL EVOLUTION (ABLATED STEP) ---
        # Evolved_flat = self.P(*M_flat) <--- SKIPPED
        
        # The aggregated state (M_flat) IS the evolved state (E_raw_evolved_flat)
        E_raw_evolved_flat = M_flat 
        
        # --- E. POST-DYNAMICS CHANNEL FUSION (CFB_dec) ---
        # CFB_dec now refines the raw aggregated M state.
        Evolved_polished = self.CFB_dec(E_raw_evolved_flat)
        
        E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = Evolved_polished

        # --- F. DECODING & RESHAPING ---
        out_frames_pred = self.D1(*Evolved_polished)
        
        predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        targets = input_clips[:, :, 1:, :, :]
        
        return predictions, targets, I0_hat, I0_gt, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved

    