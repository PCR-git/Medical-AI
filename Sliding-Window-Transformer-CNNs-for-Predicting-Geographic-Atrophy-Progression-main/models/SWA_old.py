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

# --- Lightweight Spatial Mixer ---
# class SpatioTemporalGatedMixer(nn.Module):
#     """
#     Introduces local spatial inductive bias (convolution) within the
#     Transformer block to stabilize features after the Attention pass.
#     """
#     def __init__(self, channels, kernel_size=3):
#         super().__init__()
#         # Use a depthwise separable convolution for efficiency and spatial mixing
#         self.conv = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=channels),
#             nn.GELU(),
#             nn.Conv2d(channels, channels, kernel_size=1), # Pointwise projection
#             nn.BatchNorm2d(channels)
#         )
#         self.norm = nn.LayerNorm(channels) # Pre-Norm for stability

#     def forward(self, x_tokens, H, W):
#         """Input x_tokens is (B*Axial_Size, L, C)."""
#         B_A, L, C = x_tokens.shape
#         # Reshape to (B*A, C, H, W) for 2D convolution
#         x_4d = x_tokens.reshape(-1, H, W, C).permute(0, 3, 1, 2)
        
#         # Convolutional Mixing
#         x_conv = self.conv(x_4d)
        
#         # Reshape back to tokens for FFN: (B*A, L, C)
#         x_out_tokens = x_conv.permute(0, 2, 3, 1).flatten(1, 2)
        
#         # Residual connection around the mixer (x_tokens + x_out_tokens)
#         return self.norm(x_tokens + x_out_tokens)
    
class SpatioTemporalGatedMixer(nn.Module):
    """
    Introduces local spatial inductive bias (convolution) within the Transformer block.
    Uses a learnable scalar gate (mixer_weight) to adaptively control the fusion 
    of local convolutional features with the globally-attended residual input.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        
        # Core convolutional path: Depthwise separable convolution for efficiency
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1), # Pointwise projection
            nn.BatchNorm2d(channels)
        )
        self.norm = nn.LayerNorm(channels) # Final normalization
        
        # --- Learnable Scalar Gate for Adaptive Fusion ---
        # Initialized near 0.5 to allow for both convolution and attention input
        self.mixer_weight = nn.Parameter(torch.ones(1) * 0.5) 

    def forward(self, x_tokens, D1, D2):
        """
        Args:
            x_tokens (Tensor): Input tokens from Attention path (B*Axial_Size, L, C).
            D1 (int): Dimension 1 (e.g., Time, T).
            D2 (int): Dimension 2 (e.g., Spatial, W or H).
            
        Note: L = D1 * D2.
        """
        B_A, L, C = x_tokens.shape
        T_dim, S_dim = D1, D2

        # 1. Reshape to 4D CNN format (B*Axial_Size, C, D1, D2)
        # This treats the T x S slice as a 2D image for local mixing.
        x_4d_reshaped = x_tokens.reshape(B_A, T_dim, S_dim, C).permute(0, 3, 1, 2)
        
        # 2. Convolutional Mixing
        x_conv = self.conv(x_4d_reshaped)
        
        # 3. Reshape back to tokens (B*Axial_Size, L, C)
        x_out_tokens = x_conv.permute(0, 2, 3, 1).flatten(1, 2)
        
        # 4. --- Apply Gated Fusion ---
        # Fused = Input_Residual + (Learned_Weight * Convolutional_Output)
        fused_output = x_tokens + (self.mixer_weight * x_out_tokens)
        
        # 5. Final output with LayerNorm
        return self.norm(fused_output)
    
# ---------------------------------------------------------------------------

# class  AxialTemporalSWAInterleavedLayer(nn.Module):
#     """
#     Enhanced SWA layer with an added SpatioTemporalGatedMixer after 
#     each Attention block to reintroduce local convolutional bias.
#     """
#     def __init__(self, in_channels, nhead, dim_feedforward):
#         super().__init__()
#         # SWA Attention Blocks
#         self.attn_tw = RoPEMultiheadAttention(in_channels, nhead, dropout=0, batch_first=True)
#         self.norm_tw = nn.LayerNorm(in_channels)
#         self.attn_th = RoPEMultiheadAttention(in_channels, nhead, dropout=0, batch_first=True)
#         self.norm_th = nn.LayerNorm(in_channels)
        
#         # INSERTED: Mixer blocks after each Attention pass
#         self.mixer_tw = SpatioTemporalGatedMixer(in_channels)
#         self.mixer_th = SpatioTemporalGatedMixer(in_channels)

#         self.ffn = nn.Sequential(
#             nn.Linear(in_channels, dim_feedforward),
#             nn.GELU(),
#             nn.Linear(dim_feedforward, in_channels),
#             nn.Dropout(0)
#         )
#         self.norm_ffn = nn.LayerNorm(in_channels)
#         # self.apply(init_weights) # Assuming init_weights is available

#     def forward(self, input_tokens, B, T, H, W):
#         C = input_tokens.shape[-1]
        
#         # 1. Time-Width Attention (TW)
#         tokens_tw = input_tokens.reshape(B, H, W, T, C).permute(0, 1, 3, 2, 4).reshape(B * H, T * W, C)
#         # NOTE: SWA originally uses tokens_tw, tokens_tw, tokens_tw for MHA call (self-attention)
#         attn_out_tw, _ = self.attn_tw(tokens_tw, tokens_tw, tokens_tw)
        
#         # Residual 1 (Attention)
#         attn_out_tw_res = tokens_tw + attn_out_tw
        
#         # --- MIXING STEP 1 ---
#         # Add local spatial information via convolution (Mixer)
#         mixer_out_tw = self.mixer_tw(attn_out_tw_res, T, W) 
        
#         # Residual 2 (Mixer) + LayerNorm
#         attn_out_tw_norm = self.norm_tw(attn_out_tw_res + mixer_out_tw)
#         interim_tokens = attn_out_tw_norm.reshape(B, H, T, W, C).permute(0, 2, 1, 3, 4)

#         # 2. Time-Height Attention (TH)
#         tokens_th = interim_tokens.permute(0, 3, 1, 2, 4).reshape(B * W, T * H, C)
#         attn_out_th, _ = self.attn_th(tokens_th, tokens_th, tokens_th)
        
#         # Residual 1 (Attention)
#         attn_out_th_res = tokens_th + attn_out_th
        
#         # --- MIXING STEP 2 ---
#         # Add local spatial information via convolution (Mixer)
#         mixer_out_th = self.mixer_th(attn_out_th_res, T, H)
        
#         # Residual 2 (Mixer) + LayerNorm
#         attn_out_th_norm = self.norm_th(attn_out_th_res + mixer_out_th)
        
#         final_tokens = attn_out_th_norm.reshape(B, W, T, H, C).permute(0, 2, 3, 1, 4) 
        
#         # 3. FFN
#         final_tokens_flat = final_tokens.reshape(B, T * H * W, C)
#         ffn_out = self.ffn(final_tokens_flat)
        
#         # Residual 3 (FFN)
#         output_tokens = self.norm_ffn(final_tokens_flat + ffn_out)
#         return output_tokens.reshape(B, T, H, W, C)

class AxialTemporalSWAInterleavedLayer(nn.Module):
    """
    Enhanced SWA layer with an added robust SpatioTemporalGatedMixer after 
    each Attention block to reintroduce local convolutional bias.
    """
    def __init__(self, in_channels, nhead, dim_feedforward):
        super().__init__()
        # SWA Attention Blocks
        self.attn_tw = RoPEMultiheadAttention(in_channels, nhead, dropout=0, batch_first=True)
        self.norm_tw = nn.LayerNorm(in_channels)
        self.attn_th = RoPEMultiheadAttention(in_channels, nhead, dropout=0, batch_first=True)
        self.norm_th = nn.LayerNorm(in_channels)
        
        # INSERTED: Mixer blocks after each Attention pass (Uses the strengthened Mixer)
        self.mixer_tw = SpatioTemporalGatedMixer(in_channels)
        self.mixer_th = SpatioTemporalGatedMixer(in_channels)

        self.ffn = nn.Sequential(
            nn.Linear(in_channels, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, in_channels),
            nn.Dropout(0)
        )
        self.norm_ffn = nn.LayerNorm(in_channels)
        # self.apply(init_weights) 

    def forward(self, input_tokens, B, T, H, W):
        C = input_tokens.shape[-1]
        
        # 1. Time-Width Attention (TW) - Axial on H
        tokens_tw = input_tokens.reshape(B, H, W, T, C).permute(0, 1, 3, 2, 4).reshape(B * H, T * W, C)
        attn_out_tw, _ = self.attn_tw(tokens_tw, tokens_tw, tokens_tw)
        
        # Residual 1 (Attention)
        attn_out_tw_res = tokens_tw + attn_out_tw
        
        # --- MIXING STEP 1 (Strengthened Local Bias) ---
        # Note: The mixer is applied on the B*H axis, with L=T*W tokens.
        # The H and W for the mixer need to reflect the inner dimension (T) and the slice dimension (W).
        mixer_out_tw = self.mixer_tw(attn_out_tw_res, T, W)  # Mixer now works with T, W as spatial dimensions
        
        # Residual 2 (Mixer) + LayerNorm
        # NOTE: The definition of the Mixer already includes the residual addition and LayerNorm
        # We simplify the normalization here based on the mixer's internal implementation logic.
        attn_out_tw_norm = mixer_out_tw # Mixer returns normalized token embedding
        
        interim_tokens = attn_out_tw_norm.reshape(B, H, T, W, C).permute(0, 2, 1, 3, 4)

        # 2. Time-Height Attention (TH) - Axial on W
        tokens_th = interim_tokens.permute(0, 3, 1, 2, 4).reshape(B * W, T * H, C)
        attn_out_th, _ = self.attn_th(tokens_th, tokens_th, tokens_th)
        
        # Residual 1 (Attention)
        attn_out_th_res = tokens_th + attn_out_th
        
        # --- MIXING STEP 2 (Strengthened Local Bias) ---
        # Note: The mixer is applied on the B*W axis, with L=T*H tokens.
        mixer_out_th = self.mixer_th(attn_out_th_res, T, H) # Mixer works with T, H as spatial dimensions
        
        # Residual 2 (Mixer) + LayerNorm
        attn_out_th_norm = mixer_out_th # Mixer returns normalized token embedding
        
        final_tokens = attn_out_th_norm.reshape(B, W, T, H, C).permute(0, 2, 3, 1, 4)  # Shape (B, T, H, W, C)
        
        # 3. FFN
        final_tokens_flat = final_tokens.reshape(B, T * H * W, C)
        ffn_out = self.ffn(final_tokens_flat)
        
        # Residual 3 (FFN)
        output_tokens = self.norm_ffn(final_tokens_flat + ffn_out)
        return output_tokens.reshape(B, T, H, W, C)
    
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