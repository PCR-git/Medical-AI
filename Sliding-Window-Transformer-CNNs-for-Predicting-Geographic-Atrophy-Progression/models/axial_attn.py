import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# Assuming the following imports are correctly defined in sibling modules
from models import count_parameters, init_weights
from .autoencoder import RoPEMultiheadAttention, RoPETransformerEncoderLayer
from .autoencoder import Unet_Enc, Unet_Dec, ResidualBlock
from .model_utils import init_weights, count_parameters
from .DynNet import DynNet, UPredNet
from .CFB import FusionBlockBottleneck, ChannelFusionBlock
from .SWA import LocalSpatioTemporalMixer, SpatioTemporalGatedMixer
# CRITICAL ASSUMPTION: The following RoPE utilities are available in the import path.
from .autoencoder import rotate_half, RotaryPositionalEmbedding 

# --- GLOBAL CONSTANTS (Assumed from architecture description) ---
BASE_CHANNELS = 16

# --- HELPER FUNCTIONS FOR CAUSALITY ---

def create_causal_mask(L, device):
    """Creates a basic upper-triangular mask."""
    mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
    return mask
    
def create_block_causal_mask(L, T, device):
    """
    Creates a mask that enforces causality only along the temporal dimension,
    allowing full attention within the same time step (spatial block).
    L: Total sequence length (T * H * W)
    T: Number of time steps (e.g., 3)
    """
    if L % T != 0:
        # Fallback to standard causal mask if dimensions don't align perfectly
        return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        
    S = L // T # Spatial block size (e.g., H*W for global)
    
    # temporal_causal_mask: T x T mask where future blocks are forbidden
    temporal_causal_mask = torch.triu(torch.ones((T, T), device=device), diagonal=1)
    
    # Block mask tiles the temporal mask over spatial blocks (S x S)
    block_mask = torch.kron(temporal_causal_mask, torch.ones((S, S), device=device))
    return block_mask.bool()

# --- BASE ATTENTION MODULE (AxialMultiheadAttention - RENAMED with Integrated RoPE) ---
class AxialMultiheadAttention(nn.Module):
    """ 
    Axial Multihead Attention with integrated RoPE.
    RoPE is applied to Q and K before matrix multiplication to preserve the positional prior.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Add RoPE component
        self.rope_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len=4096)
        
    def _reshape_heads(self, t):
        N, L, D = t.shape
        # Reshape to [N*H, L, D_H]
        return t.reshape(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(N * self.num_heads, L, self.head_dim)

    def forward(self, x, attn_mask=None): # Simplified API for mask passing
        N, L, D = x.shape
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # --- ROPE APPLICATION ---
        rot_emb_cos_sin = self.rope_emb(L, x.device) 
        cos, sin = rot_emb_cos_sin.chunk(2, dim=0)
        # Reshape for broadcasting [1, 1, L, D_H]
        cos = cos.unsqueeze(1) 
        sin = sin.unsqueeze(1)

        def apply_rope(t):
            # Reshape from [N, L, D] to [N, H, L, D_H]
            t_reshaped = t.reshape(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            # Apply RoPE: t_rot = t * cos + rotate_half(t) * sin
            t_rot = (t_reshaped * cos) + (rotate_half(t_reshaped) * sin)
            # Reshape back to [N, L, D] 
            return t_rot.permute(0, 2, 1, 3).reshape(N, L, D)

        q = apply_rope(q)
        k = apply_rope(k)
        # V remains without RoPE
        # --- END ROPE APPLICATION ---

        q_head = self._reshape_heads(q)
        k_head = self._reshape_heads(k)
        v_head = self._reshape_heads(v)
        
        attn_matrix = torch.matmul(q_head * self.scaling, k_head.transpose(-2, -1))
        
        if attn_mask is not None:
            attn_matrix.masked_fill_(attn_mask.unsqueeze(0), float('-inf'))
        
        attn_output_weights = F.softmax(attn_matrix, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        
        attn_output = torch.matmul(attn_output_weights, v_head)
        
        attn_output_reshaped = attn_output.reshape(N, self.num_heads, L, self.head_dim).permute(0, 2, 1, 3).flatten(start_dim=-2)
        output = self.out_proj(attn_output_reshaped)
        
        return output, attn_output_weights.mean(dim=1)


# --- L5 GLOBAL CAUSAL INTEGRATOR (for the single-pass baseline) ---
class GlobalCausalIntegrator(nn.Module):
    """
    Implements a single pass of Full Global Causal Attention for the L5 Bottleneck.
    This module must be RoPE-enabled to match the main SWAU_Net design.
    """
    def __init__(self, in_channels, nhead, dim_feedforward):
        super().__init__()
        # Uses RoPETransformerEncoderLayer for the global attention
        # NOTE: RoPETransformerEncoderLayer is assumed to correctly apply the causal mask
        self.attn_layer = RoPETransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0,
            batch_first=True
        )

    def forward(self, stacked_features_4d):
        B, T, C, H, W = stacked_features_4d.shape
        L = T * H * W
        
        # 1. Flatten into a single token sequence: [B, L, C]
        # Permute needed for correct spatial/temporal ordering before flattening
        input_tokens = stacked_features_4d.permute(0, 1, 3, 4, 2).reshape(B, L, C)

        # 2. Create Global Causal Mask
        causal_mask = create_block_causal_mask(L, T, input_tokens.device)
        
        # 3. Perform Full Global Causal Attention 
        output_tokens = self.attn_layer(input_tokens, attn_mask=causal_mask)
        
        # 4. Reshape back to (B, T, C, H, W)
        output_tokens_4d = output_tokens.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)
        
        return output_tokens_4d


# --- L4 AXIAL CAUSAL ATTENTION COMPONENTS ---

class StandardAxialInterleavedLayer(nn.Module):
    """ 
    Implements a single layer of interleaved causal axial attention.
    Now uses the AxialMultiheadAttention (with integrated RoPE) for L4.
    """
    def __init__(self, in_channels, nhead, dim_feedforward):
        super().__init__()
        # *** Using AxialMultiheadAttention (with integrated RoPE) ***
        self.attn_tw = AxialMultiheadAttention(in_channels, nhead, dropout=0)
        self.norm_tw = nn.LayerNorm(in_channels)
        self.attn_th = AxialMultiheadAttention(in_channels, nhead, dropout=0)
        self.norm_th = nn.LayerNorm(in_channels)
        
        # Mixer blocks (SpatioTemporalGatedMixer is assumed to be defined in .SWA)
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
        
        # --- 1. Time-Width Attention (TW) - Axial on H ---
        tokens_tw = input_tokens.reshape(B, T, H, W, C).permute(0, 2, 1, 3, 4).reshape(B * H, T * W, C)
        original_input_tw = tokens_tw
        
        L_tw = tokens_tw.size(1) 
        causal_mask_tw = create_block_causal_mask(L_tw, T, tokens_tw.device)
        
        # Perform Attention (Attn_Out) - Output is the computed delta
        attn_out_tw, _ = self.attn_tw(tokens_tw, attn_mask=causal_mask_tw)
        
        # Mixer Step: Mixer calculates (Attn_Out + Gated Conv Delta)
        mixer_out_tw_update = self.mixer_tw(attn_out_tw, T, W)
        
        # Final LayerNorm: Norm(Original Input + Total Update Delta)
        attn_out_tw_norm = self.norm_tw(original_input_tw + mixer_out_tw_update)
        
        interim_tokens = attn_out_tw_norm.reshape(B, H, T, W, C).permute(0, 2, 1, 3, 4)

        # --- 2. Time-Height Attention (TH) - Axial on W ---
        tokens_th = interim_tokens.permute(0, 3, 2, 1, 4).reshape(B * W, T * H, C)
        original_input_th = tokens_th

        L_th = tokens_th.size(1) 
        causal_mask_th = create_block_causal_mask(L_th, T, tokens_th.device)
        
        # Perform Attention (Attn_Out)
        attn_out_th, _ = self.attn_th(tokens_th, attn_mask=causal_mask_th)
        
        # Mixer Step
        mixer_out_th_update = self.mixer_th(attn_out_th, T, H)
        
        # Final LayerNorm
        attn_out_th_norm = self.norm_th(original_input_th + mixer_out_th_update)
        
        final_tokens = attn_out_th_norm.reshape(B, W, T, H, C).permute(0, 2, 3, 1, 4)
        
        # 3. FFN (Standard Residual)
        final_tokens_flat = final_tokens.reshape(B, T * H * W, C)
        ffn_out = self.ffn(final_tokens_flat)
        output_tokens = self.norm_ffn(final_tokens_flat + ffn_out)
        
        return output_tokens.reshape(B, T, H, W, C)
    
class StandardAxialIntegrator(nn.Module):
    """ 
    Wrapper for StandardAxialInterleavedLayer, performing multiple layers.
    Used for L4 in the single-pass baseline.
    """
    def __init__(self, in_channels, nhead, dim_feedforward, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            StandardAxialInterleavedLayer(in_channels, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
    
    def forward(self, stacked_features_4d):
        # stacked_features_4d shape: (B, T, C, H, W)
        B, T, C, H, W = stacked_features_4d.shape
        # Convert to tokens: (B, T, H, W, C)
        tokens = stacked_features_4d.permute(0, 1, 3, 4, 2)

        for layer in self.layers:
            tokens = layer(tokens, B, T, H, W)
            
        # Output the full integrated feature sequence (B, T, H, W, C) -> (B, T, C, H, W)
        return tokens.permute(0, 1, 4, 2, 3)

# --- BASELINE FEATURE AGGREGATOR ---

class StandardAxialFeatureAggregator(nn.Module):
    """ 
    The core Aggregation module for the Causal Axial Baseline.
    L5 and L4 use Axial Attention. L1-L3 use CNN mixers.
    """
    def __init__(self, args, base_channels=None):
        super().__init__()
        # BASE_CHANNELS and NUM_ATTN_LAYERS are assumed to be defined/available via args
        C = base_channels if base_channels is not None else 16
        NUM_ATTN_LAYERS = args.num_attn_layers 
        
        C_L2_INPUT = C; C_L2_CONCAT_CH = 3 * C_L2_INPUT; C_L2_SKIP_OUT = C
        C_L1_INPUT = C // 2; C_L1_CONCAT_CH = 3 * C_L1_INPUT; C_L1_SKIP_OUT = C // 2

        # L5 Attention (AXIAL CAUSAL INTEGRATOR)
        # This mirrors the complexity of the original SWAU-Net L5 Core (Axial)
        self.attn_bn = StandardAxialIntegrator(
            in_channels=C * 16, nhead=args.nhead, dim_feedforward=args.d_attn2, num_layers=NUM_ATTN_LAYERS
        )
        # L4 Attention (AXIAL CAUSAL INTEGRATOR)
        self.attn_l4 = StandardAxialIntegrator(
            in_channels=C * 8, nhead=args.nhead, dim_feedforward=args.d_attn2, num_layers=NUM_ATTN_LAYERS
        )
        
        # CNN Mixer (L3) - LocalSpatioTemporalMixer
        self.mixer_l3 = LocalSpatioTemporalMixer(in_channels=C * 4, kernel_size=3)
        
        # CNN Mixers (L2, L1)
        self.conv_l2 = nn.Sequential(nn.Conv2d(C_L2_CONCAT_CH, C_L2_SKIP_OUT, kernel_size=1), nn.GELU(), ResidualBlock(C_L2_SKIP_OUT, C_L2_SKIP_OUT))
        self.conv_l1 = nn.Sequential(nn.Conv2d(C_L1_CONCAT_CH, C_L1_SKIP_OUT, kernel_size=1), nn.GELU(), ResidualBlock(C_L1_SKIP_OUT, C_L1_SKIP_OUT))
        self.apply(init_weights)  

    def forward(self, F0_refined, F1_refined, F2_refined):
        # Fx_refined are lists of 5 features (f1, f2, f3, f4, bn) for a single time step (B samples).
        
        # 1. Stack all features into (B, T=3, C, H, W) format
        bn_stacked = torch.stack([F0_refined[4], F1_refined[4], F2_refined[4]], dim=1)
        f4_stacked = torch.stack([F0_refined[3], F1_refined[3], F2_refined[3]], dim=1)
        
        # 2. Attention Aggregation (BN and L4) - Output is (B, T, C, H, W)
        M_bn_stacked = self.attn_bn(bn_stacked)
        M4_stacked = self.attn_l4(f4_stacked)

        # 3. Macro-Residual Addition for Attention Layers
        M_bn_stacked = M_bn_stacked + bn_stacked
        M4_stacked = M4_stacked + f4_stacked
        
        # 4. CNN Mixer Aggregation (L3, L2, L1) - These still use CNNs across the 3 frames
        f3_t_2, f3_t_1, f3_t = F0_refined[2], F1_refined[2], F2_refined[2]
        f2_t_2, f2_t_1, f2_t = F0_refined[1], F1_refined[1], F2_refined[1]
        f1_t_2, f1_t_1, f1_t = F0_refined[0], F1_refined[0], F2_refined[0]

        # L3 uses LocalSpatioTemporalMixer (output is only M_t, includes F_t residual internally)
        M3_t = self.mixer_l3(f3_t_2, f3_t_1, f3_t)
        
        # L2, L1 use concatenation + CNN (requires manual F_t residual for stability)
        f2_cat = torch.cat([f2_t_2, f2_t_1, f2_t], dim=1); M2_t = self.conv_l2(f2_cat)
        M2_t = M2_t + f2_t 
        
        f1_cat = torch.cat([f1_t_2, f1_t_1, f1_t], dim=1); M1_t = self.conv_l1(f1_cat)
        M1_t = M1_t + f1_t 
        
        # 5. Prepare Output M_flat for DynNet (B*T, C, H, W)
        
        # L4, BN: Flatten the full (B, T, C, H, W) sequence
        M_bn_cat = M_bn_stacked.reshape(-1, *M_bn_stacked.shape[2:])
        M4_cat = M4_stacked.reshape(-1, *M4_stacked.shape[2:])

        # L1, L2, L3: Replicate the current time step M_t across T=3
        M3_cat = torch.cat([M3_t] * 3, dim=0)
        M2_cat = torch.cat([M2_t] * 3, dim=0)
        M1_cat = torch.cat([M1_t] * 3, dim=0)
        
        return M1_cat, M2_cat, M3_cat, M4_cat, M_bn_cat

# class StandardAxialFeatureAggregator(nn.Module):
#     """ 
#     The core Aggregation module for the Causal Axial Baseline.
#     Performs a single causal pass over T=3 for L4/L5, and simple mixing for L1-L3.
#     """
#     def __init__(self, args, base_channels=None):
#         super().__init__()
#         C = base_channels if base_channels is not None else BASE_CHANNELS
#         NUM_ATTN_LAYERS = args.num_attn_layers # Typically 2 or 3
        
#         C_L2_INPUT = C; C_L2_CONCAT_CH = 3 * C_L2_INPUT; C_L2_SKIP_OUT = C
#         C_L1_INPUT = C // 2; C_L1_CONCAT_CH = 3 * C_L1_INPUT; C_L1_SKIP_OUT = C // 2

#         # L5 Attention (FULL GLOBAL CAUSAL INTEGRATOR)
#         self.attn_bn = GlobalCausalIntegrator(
#             in_channels=C * 16, nhead=args.nhead, dim_feedforward=args.d_attn2
#         )
#         # L4 Attention (AXIAL CAUSAL INTEGRATOR)
#         self.attn_l4 = StandardAxialIntegrator(
#             in_channels=C * 8, nhead=args.nhead, dim_feedforward=args.d_attn2, num_layers=NUM_ATTN_LAYERS
#         )
        
#         # CNN Mixer (L3) - LocalSpatioTemporalMixer (assumed defined in .SWA)
#         self.mixer_l3 = LocalSpatioTemporalMixer(in_channels=C * 4, kernel_size=3)
        
#         # CNN Mixers (L2, L1)
#         self.conv_l2 = nn.Sequential(nn.Conv2d(C_L2_CONCAT_CH, C_L2_SKIP_OUT, kernel_size=1), nn.GELU(), ResidualBlock(C_L2_SKIP_OUT, C_L2_SKIP_OUT))
#         self.conv_l1 = nn.Sequential(nn.Conv2d(C_L1_CONCAT_CH, C_L1_SKIP_OUT, kernel_size=1), nn.GELU(), ResidualBlock(C_L1_SKIP_OUT, C_L1_SKIP_OUT))
#         self.apply(init_weights)  

#     def forward(self, F0_refined, F1_refined, F2_refined):
#         # Fx_refined are lists of 5 features (f1, f2, f3, f4, bn) for a single time step (B samples).
        
#         # 1. Stack all features into (B, T=3, C, H, W) format
#         bn_stacked = torch.stack([F0_refined[4], F1_refined[4], F2_refined[4]], dim=1)
#         f4_stacked = torch.stack([F0_refined[3], F1_refined[3], F2_refined[3]], dim=1)
        
#         # 2. Attention Aggregation (BN and L4) - Output is (B, T, C, H, W)
#         M_bn_stacked = self.attn_bn(bn_stacked)
#         M4_stacked = self.attn_l4(f4_stacked)

#         # 3. Macro-Residual Addition for Attention Layers
#         M_bn_stacked = M_bn_stacked + bn_stacked
#         M4_stacked = M4_stacked + f4_stacked
        
#         # 4. CNN Mixer Aggregation (L3, L2, L1) - These still use CNNs across the 3 frames
#         f3_t_2, f3_t_1, f3_t = F0_refined[2], F1_refined[2], F2_refined[2]
#         f2_t_2, f2_t_1, f2_t = F0_refined[1], F1_refined[1], F2_refined[1]
#         f1_t_2, f1_t_1, f1_t = F0_refined[0], F1_refined[0], F2_refined[0]

#         # L3 uses LocalSpatioTemporalMixer (output is only M_t, includes F_t residual internally)
#         M3_t = self.mixer_l3(f3_t_2, f3_t_1, f3_t)
        
#         # L2, L1 use concatenation + CNN (requires manual F_t residual for stability)
#         f2_cat = torch.cat([f2_t_2, f2_t_1, f2_t], dim=1); M2_t = self.conv_l2(f2_cat)
#         M2_t = M2_t + f2_t 
        
#         f1_cat = torch.cat([f1_t_2, f1_t_1, f1_t], dim=1); M1_t = self.conv_l1(f1_cat)
#         M1_t = M1_t + f1_t 
        
#         # 5. Prepare Output M_flat for DynNet (B*T, C, H, W)
        
#         # L4, BN: Flatten the full (B, T, C, H, W) sequence
#         M_bn_cat = M_bn_stacked.reshape(-1, *M_bn_stacked.shape[2:])
#         M4_cat = M4_stacked.reshape(-1, *M4_stacked.shape[2:])

#         # L1, L2, L3: Replicate the current time step M_t across T=3
#         M3_cat = torch.cat([M3_t] * 3, dim=0)
#         M2_cat = torch.cat([M2_t] * 3, dim=0)
#         M1_cat = torch.cat([M1_t] * 3, dim=0)
        
#         return M1_cat, M2_cat, M3_cat, M4_cat, M_bn_cat

# --- MAIN MODEL WRAPPER ---

class AxialU_Net(nn.Module):
    """
    Standard Axial Causal U-Net (AxialU_Net) - The single-pass causal baseline 
    to be compared against SWAU_Net.
    """
    def __init__(self, args, img_channels=None, base_channels=None):
        super(AxialU_Net, self).__init__()
        
        if img_channels is None: img_channels = 3
        if base_channels is None: base_channels = BASE_CHANNELS
            
        # Shared Components (Encoder, DynNet, Decoder)
        # Assuming Unet_Enc and DynNet now include the L5 spatial attention updates
        self.E1 = Unet_Enc(args, img_channels, base_channels)
        self.CFB_enc = ChannelFusionBlock(base_channels)
        self.P = DynNet(args, base_channels)
        self.CFB_dec = ChannelFusionBlock(base_channels)
        self.D1 = Unet_Dec(args, img_channels, base_channels)
        
        # *** KEY DIFFERENCE: The single-pass Causal Aggregator ***
        self.Axial_Aggregator = StandardAxialFeatureAggregator(args, base_channels)
        
    def forward(self, input_clips):
        B, C, T, H, W = input_clips.shape
        T_pred = T - 1  # 3 predicted frames
        
        # A. Feature Extraction and Refinement
        I0_gt = input_clips[:, :, 0, :, :]
        input_frames_E1 = input_clips[:, :, :T_pred, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        E1_features_flat = self.E1(input_frames_E1)
        F_refined_flat = self.CFB_enc(E1_features_flat)
        
        # Unpack F features by time step 
        F0_refined = [f[:B] for f in F_refined_flat]
        F1_refined = [f[B:2*B] for f in F_refined_flat]
        F2_refined = [f[2*B:3*B] for f in F_refined_flat]
        
        # B. Anchor Reconstruction
        I0_hat = self.D1(*F0_refined)
        
        # C. Causal Axial Aggregation (Single Pass M_flat is calculated)
        M_flat = self.Axial_Aggregator(F0_refined, F1_refined, F2_refined)
        
        # D. Temporal Evolution and Decoding (Same as SWAU_Net)
        E_raw_evolved_flat = self.P(*M_flat)
        Evolved_polished = self.CFB_dec(E_raw_evolved_flat)

        E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = Evolved_polished
        out_frames_pred = self.D1(*Evolved_polished)
        
        predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        targets = input_clips[:, :, 1:, :, :]
        
        return predictions, targets, I0_hat, I0_gt, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved
    