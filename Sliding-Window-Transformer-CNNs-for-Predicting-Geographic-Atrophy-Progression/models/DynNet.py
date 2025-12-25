import torch
import torch.nn as nn
import torch.nn.functional as F 

from .autoencoder import RoPEMultiheadAttention, RoPETransformerEncoderLayer
from .autoencoder import Unet_Enc, Unet_Dec, ResidualBlock
from .model_utils import init_weights
from .CFB import FusionBlockBottleneck, ChannelFusionBlock

# ------------------------------------------------------------------------------
# DYN-NET
# ------------------------------------------------------------------------------

# class TemporalDeltaBlock(nn.Module):
#     """
#     A simplified convolutional block (non-residual) used to predict the DELTA
#     of feature evolution (Et+1 - Mt). Used for high-resolution levels (L1, L2)
#     where the delta is expected to be sparse and small.
#     """
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         # Simpler structure than ResidualBlock, focusing on mapping inputs to a delta.
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.GELU(),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.GELU(),
#         )
#         # Assuming init_weights is available
#         self.apply(init_weights) 

#     def forward(self, x):
#         return self.conv(x)


# class DynNet(nn.Module):
#     # Changed base_channels default to None to resolve NameError on module load
#     def __init__(self, args, base_channels=None):
#         super(DynNet, self).__init__()
        
#         # Resolve base_channels default inside __init__
#         C = base_channels if base_channels is not None else 16

#         self.te1_model_dim = C * 8 
#         self.te2_model_dim = C * 4 
        
#         C_BN = C * 16
        
#         # L2 FUSION INPUT: 
#         # L2 Input Feature (from SWA/Recurrence) is C=16. Upsampled Bottleneck is C_BN//8 = 32. 
#         # Total expected input to ResidualBlock is 16 + 32 = 48 channels.
#         C_L2_FUSION_IN = C + (C_BN // 8) # 16 + 32 = 48
        
#         # L1 FUSION INPUT:
#         # L1 Input Feature (from SWA/Recurrence) is C/2=8. Upsampled Bottleneck is C_BN//16 = 16.
#         # Total expected input to ResidualBlock is 8 + 16 = 24 channels.
#         C_L1_FUSION_IN = (C // 2) + (C_BN // 16) # 8 + 16 = 24

#         self.pred_bottleneck = ResidualBlock(C * 16, C * 16)
#         self.pred_feat4_attn = RoPETransformerEncoderLayer(d_model=self.te1_model_dim, nhead=args.nhead, dim_feedforward=args.d_attn2, dropout=0, batch_first=True)
#         self.pred_feat4_cnn = ResidualBlock(C * 8, C * 8)
#         self.pred_feat3_attn = RoPETransformerEncoderLayer(d_model=self.te2_model_dim, nhead=args.nhead, dim_feedforward=args.d_attn1, dropout=0, batch_first=True)
#         self.pred_feat3_cnn = ResidualBlock(C * 4, C * 4)

#         self.upsample_bn_to_L2 = nn.ConvTranspose2d(C_BN, C_BN // 8, kernel_size=8, stride=8) 
#         self.upsample_bn_to_L1 = nn.ConvTranspose2d(C_BN, C_BN // 16, kernel_size=16, stride=16) 

#         # Level 2 Prediction Head: Input size 48. Output size C*2=32 (then reduced to C=16)
#         self.pred_feat2_fused = nn.Sequential(
#             ResidualBlock(C_L2_FUSION_IN, C * 2), # Input 48 -> Output 32
#             nn.Conv2d(C * 2, C, kernel_size=1),  # Reduction 32 -> 16
#             nn.GELU()
#         )

#         # Level 1 Prediction Head: Input size 24. Output size C=16 (then reduced to C/2=8)
#         self.pred_feat1_fused = nn.Sequential(
#             ResidualBlock(C_L1_FUSION_IN, C), # Input 24 -> Output 16
#             nn.Conv2d(C, C // 2, kernel_size=1), # Reduction 16 -> 8
#             nn.GELU()
#         )

#         self.apply(init_weights)

#     def forward(self, feats1u, feats2u, feats3u, feats4u, bottleneck_4d):
#         """
#         featsXu are the M_t features aggregated by SWA (the input to DynNet)
#         """
#         N = bottleneck_4d.size(0)

#         # --- 1. Evolve Bottleneck (Temporal State) ---
#         E_bn = self.pred_bottleneck(bottleneck_4d) 
#         E_bn = E_bn + bottleneck_4d  # CRITICAL SKIP: Add input BN feature to evolved BN feature
        
#         # --- 2. Evolve Level 4 (Transformer Path) ---
#         tokens4 = feats4u.flatten(2).transpose(1, 2)
#         attn_out4 = self.pred_feat4_attn(tokens4)
#         attn_out4_4d = attn_out4.transpose(1, 2).reshape(N, self.te1_model_dim, 32, 32)
#         feats4u_pred = self.pred_feat4_cnn(feats4u + attn_out4_4d) 
#         feats4u_pred = feats4u_pred + feats4u # CRITICAL SKIP: Add input L4 feature to evolved L4 feature

#         # --- 3. Evolve Level 3 (Transformer Path) ---
#         tokens3 = feats3u.flatten(2).transpose(1, 2)
#         attn_out3 = self.pred_feat3_attn(tokens3)
#         attn_out3_4d = attn_out3.transpose(1, 2).reshape(N, self.te2_model_dim, 64, 64)
#         feats3u_pred = self.pred_feat3_cnn(feats3u + attn_out3_4d) 
#         feats3u_pred = feats3u_pred + feats3u # CRITICAL SKIP: Add input L3 feature to evolved L3 feature
        
#         # --- 4. Evolve Level 2 (Fused Path) ---
#         E_bn_L2_up = self.upsample_bn_to_L2(E_bn) 
#         fused_L2 = torch.cat([feats2u, E_bn_L2_up], dim=1) 
#         feats2u_pred = self.pred_feat2_fused(fused_L2) 
#         feats2u_pred = feats2u_pred + feats2u # CRITICAL SKIP: Add input L2 feature to evolved L2 feature

#         # --- 5. Evolve Level 1 (Fused Path) ---
#         E_bn_L1_up = self.upsample_bn_to_L1(E_bn) 
#         fused_L1 = torch.cat([feats1u, E_bn_L1_up], dim=1) 
#         feats1u_pred = self.pred_feat1_fused(fused_L1) 
#         feats1u_pred = feats1u_pred + feats1u # CRITICAL SKIP: Add input L1 feature to evolved L1 feature
        
#         # NOTE: DynNet outputs [8, 16, 64, 128, 256] channels, which matches 
#         # the Encoder's required input structure for recurrence.
#         return feats1u_pred, feats2u_pred, feats3u_pred, feats4u_pred, E_bn


# class DynNet(nn.Module):
#     """
#     A simplified DynNet using only ResidualBlocks for feature evolution,
#     designed to be comparable in complexity to a single step of ConvLSTM.
#     It takes M_t features and evolves them to E_t+1 features.
#     """
#     def __init__(self, args, base_channels=16):
#         super(DynNet, self).__init__()
#         C = base_channels

#         C_BN = C * 16

#         # L2 FUSION INPUT: L2 Feature (C) + Upsampled BN (C_BN // 8) = 48
#         C_L2_FUSION_IN = C + (C_BN // 8) 
#         # L1 FUSION INPUT: L1 Feature (C//2) + Upsampled BN (C_BN // 16) = 24
#         C_L1_FUSION_IN = (C // 2) + (C_BN // 16) 

#         # --- Multiscale Feature Evolution Heads (Conv-only) ---
#         # 1. Bottleneck: Replaces pred_bottleneck with a simple ResidualBlock.
#         self.pred_bottleneck = ResidualBlock(C * 16, C * 16)

#         # 2. Level 4: Replaces Transformer with a ResidualBlock.
#         self.pred_feat4_cnn = ResidualBlock(C * 8, C * 8)
        
#         # 3. Level 3: Replaces Transformer with a ResidualBlock.
#         self.pred_feat3_cnn = ResidualBlock(C * 4, C * 4)

#         # --- Upsampling for Shallow Fusion ---
#         self.upsample_bn_to_L2 = nn.ConvTranspose2d(C_BN, C_BN // 8, kernel_size=8, stride=8) 
#         self.upsample_bn_to_L1 = nn.ConvTranspose2d(C_BN, C_BN // 16, kernel_size=16, stride=16) 

#         # 4. Level 2 Prediction Head (Fused with Upsampled BN)
#         self.pred_feat2_fused = nn.Sequential(
#             ResidualBlock(C_L2_FUSION_IN, C * 2),
#             nn.Conv2d(C * 2, C, kernel_size=1), 
#             nn.GELU()
#         )

#         # 5. Level 1 Prediction Head (Fused with Upsampled BN)
#         self.pred_feat1_fused = nn.Sequential(
#             ResidualBlock(C_L1_FUSION_IN, C),
#             nn.Conv2d(C, C // 2, kernel_size=1),
#             nn.GELU()
#         )

#         # Assuming init_weights is available
#         try:
#             self.apply(init_weights)
#         except NameError:
#             print("Warning: init_weights not found. Skipping initialization.")


#     def forward(self, feats1u, feats2u, feats3u, feats4u, bottleneck_4d):
#         """
#         featsXu are the M_t features (input from SWA/Recurrence)
#         Outputs are the E_t+1 features.
#         """
        
#         # --- 1. Evolve Bottleneck (Deepest Layer) ---
#         # E_bn = f(M_bn) + M_bn (Residual connection ensures feature persistence)
#         E_bn = self.pred_bottleneck(bottleneck_4d)
#         E_bn = E_bn + bottleneck_4d

#         # --- 2. Evolve Level 4 (CNN Path) ---
#         # Remove Transformer layers and only use CNN
#         feats4u_pred = self.pred_feat4_cnn(feats4u) 
#         feats4u_pred = feats4u_pred + feats4u

#         # --- 3. Evolve Level 3 (CNN Path) ---
#         # Remove Transformer layers and only use CNN
#         feats3u_pred = self.pred_feat3_cnn(feats3u)
#         feats3u_pred = feats3u_pred + feats3u

#         # --- 4. Evolve Level 2 (Fused Path) ---
#         E_bn_L2_up = self.upsample_bn_to_L2(E_bn)
#         fused_L2 = torch.cat([feats2u, E_bn_L2_up], dim=1)
#         feats2u_pred = self.pred_feat2_fused(fused_L2)
#         feats2u_pred = feats2u_pred + feats2u

#         # --- 5. Evolve Level 1 (Fused Path) ---
#         E_bn_L1_up = self.upsample_bn_to_L1(E_bn)
#         fused_L1 = torch.cat([feats1u, E_bn_L1_up], dim=1)
#         feats1u_pred = self.pred_feat1_fused(fused_L1)
#         feats1u_pred = feats1u_pred + feats1u
        
#         return feats1u_pred, feats2u_pred, feats3u_pred, feats4u_pred, E_bn
    
# ------------------------------------------------------------------------------ 

# class DynNet(nn.Module):
#     # Changed base_channels default to None to resolve NameError on module load
#     def __init__(self, args, base_channels=None):
#         super(DynNet, self).__init__()
        
#         # Resolve base_channels default inside __init__
#         C = base_channels if base_channels is not None else 16

#         self.te1_model_dim = C * 8  
#         self.te2_model_dim = C * 4  
        
#         C_BN = C * 16
        
#         # L2 FUSION INPUT: 16 + 32 = 48
#         C_L2_FUSION_IN = C + (C_BN // 8) 
        
#         # L1 FUSION INPUT: 8 + 16 = 24
#         C_L1_FUSION_IN = (C // 2) + (C_BN // 16) 

#         # --- Bottleneck/Deep Layers (Full State Evolution) ---
#         self.pred_bottleneck = ResidualBlock(C * 16, C * 16)
        
#         # L4 Prediction Head (Attention Delta)
#         self.pred_feat4_attn = RoPETransformerEncoderLayer(d_model=self.te1_model_dim, nhead=args.nhead, dim_feedforward=args.d_attn2, dropout=0, batch_first=True)
#         self.pred_feat4_cnn = ResidualBlock(C * 8, C * 8)
        
#         # L3 Prediction Head (Attention Delta)
#         self.pred_feat3_attn = RoPETransformerEncoderLayer(d_model=self.te2_model_dim, nhead=args.nhead, dim_feedforward=args.d_attn1, dropout=0, batch_first=True)
#         self.pred_feat3_cnn = ResidualBlock(C * 4, C * 4)

#         # --- Upsampling for Fused Paths ---
#         self.upsample_bn_to_L2 = nn.ConvTranspose2d(C_BN, C_BN // 8, kernel_size=8, stride=8)  
#         self.upsample_bn_to_L1 = nn.ConvTranspose2d(C_BN, C_BN // 16, kernel_size=16, stride=16)  

#         # --- L2 Prediction Head (DELTA PREDICTION) ---
#         # Predicts the DELTA features for L2: Input 48 -> Output 16 (C)
#         self.pred_feat2_fused = nn.Sequential(
#             ResidualBlock(C_L2_FUSION_IN, C * 2), # Input 48 -> Output 32
#             nn.Conv2d(C * 2, C, kernel_size=1),  # Reduction 32 -> 16 (The Delta)
#             nn.GELU()
#         )

#         # --- L1 Prediction Head (DELTA PREDICTION) ---
#         # Predicts the DELTA features for L1: Input 24 -> Output 8 (C/2)
#         self.pred_feat1_fused = nn.Sequential(
#             ResidualBlock(C_L1_FUSION_IN, C), # Input 24 -> Output 16
#             nn.Conv2d(C, C // 2, kernel_size=1), # Reduction 16 -> 8 (The Delta)
#             nn.GELU()
#         )

#         self.apply(init_weights)

#     def forward(self, feats1u, feats2u, feats3u, feats4u, bottleneck_4d):
#         """
#         featsXu are the M_t features aggregated by SWA (the input to DynNet)
#         Outputs Evolved features E_t+1 = M_t + Delta_t
#         """
#         N = bottleneck_4d.size(0)

#         # --- 1. Evolve Bottleneck (Temporal State) ---
#         # E_bn is the full evolved state.
#         E_bn = self.pred_bottleneck(bottleneck_4d)  
#         E_bn = E_bn + bottleneck_4d  # CRITICAL SKIP: Add input BN feature to evolved BN feature
        
#         # --- 2. Evolve Level 4 (Attention Delta) ---
#         tokens4 = feats4u.flatten(2).transpose(1, 2)
#         # attn_out4 is the residual update computed by the transformer
#         attn_delta4 = self.pred_feat4_attn(tokens4) 
#         attn_delta4_4d = attn_delta4.transpose(1, 2).reshape(N, self.te1_model_dim, 32, 32)
        
#         # Evolved L4 = Input M4 + (CNN refinement of Attention Delta)
#         feats4u_evolved = self.pred_feat4_cnn(feats4u + attn_delta4_4d)  
#         feats4u_evolved = feats4u_evolved + feats4u # CRITICAL SKIP: Add input L4 feature to evolved L4 feature

#         # --- 3. Evolve Level 3 (Attention Delta) ---
#         tokens3 = feats3u.flatten(2).transpose(1, 2)
#         attn_delta3 = self.pred_feat3_attn(tokens3)
#         attn_delta3_4d = attn_delta3.transpose(1, 2).reshape(N, self.te2_model_dim, 64, 64)
        
#         # Evolved L3 = Input M3 + (CNN refinement of Attention Delta)
#         feats3u_evolved = self.pred_feat3_cnn(feats3u + attn_delta3_4d)  
#         feats3u_evolved = feats3u_evolved + feats3u # CRITICAL SKIP: Add input L3 feature to evolved L3 feature
        
#         # --- 4. Evolve Level 2 (Fused Path - DELTA PREDICTION) ---
#         E_bn_L2_up = self.upsample_bn_to_L2(E_bn)  
#         fused_L2 = torch.cat([feats2u, E_bn_L2_up], dim=1)  
        
#         # Predict Delta from fused input
#         delta_L2 = self.pred_feat2_fused(fused_L2)
#         # Evolved L2 = Input M2 + Predicted Delta
#         feats2u_evolved = feats2u + delta_L2 # This enforces sparse update
#         feats2u_evolved = feats2u_evolved + feats2u # Final Residual Block (Main Skip)
        
#         # --- 5. Evolve Level 1 (Fused Path - DELTA PREDICTION) ---
#         E_bn_L1_up = self.upsample_bn_to_L1(E_bn)  
#         fused_L1 = torch.cat([feats1u, E_bn_L1_up], dim=1)  
        
#         # Predict Delta from fused input
#         delta_L1 = self.pred_feat1_fused(fused_L1) 
#         # Evolved L1 = Input M1 + Predicted Delta
#         feats1u_evolved = feats1u + delta_L1 # This enforces sparse update
#         feats1u_evolved = feats1u_evolved + feats1u # Final Residual Block (Main Skip)
        
#         return feats1u_evolved, feats2u_evolved, feats3u_evolved, feats4u_evolved, E_bn

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
    
class DynNet(nn.Module):
    def __init__(self, args, base_channels=None):
        super(DynNet, self).__init__()
        
        C = base_channels if base_channels is not None else 16

        self.te1_model_dim = C * 8  
        self.te2_model_dim = C * 4  
        
        C_BN = C * 16 # e.g., 256
        
        # L2 FUSION INPUT: 16 + 32 = 48
        C_L2_FUSION_IN = C + (C_BN // 8) 
        
        # L1 FUSION INPUT: 8 + 16 = 24
        C_L1_FUSION_IN = (C // 2) + (C_BN // 16) 

        # --- Core Evolution Heads ---
        self.pred_bottleneck = ResidualBlock(C * 16, C * 16)
        
        # L4 Prediction Head (Attention Delta)
        self.pred_feat4_attn = RoPETransformerEncoderLayer(d_model=self.te1_model_dim, nhead=args.nhead, dim_feedforward=args.d_attn2, dropout=0, batch_first=True)
        self.pred_feat4_cnn = ResidualBlock(C * 8, C * 8)
        
        # L3 Prediction Head (Attention Delta)
        self.pred_feat3_attn = RoPETransformerEncoderLayer(d_model=self.te2_model_dim, nhead=args.nhead, dim_feedforward=args.d_attn1, dropout=0, batch_first=True)
        self.pred_feat3_cnn = ResidualBlock(C * 4, C * 4)

        # --- EXISTING: Upsampling for Fused Paths (L1 & L2) ---
        self.upsample_bn_to_L2 = nn.ConvTranspose2d(C_BN, C_BN // 8, kernel_size=8, stride=8)  
        self.upsample_bn_to_L1 = nn.ConvTranspose2d(C_BN, C_BN // 16, kernel_size=16, stride=16)  

        # --- MODIFIED: Low-Cost Bottleneck Projection for L3 & L4 ---
        # 1x1 convolutions for channel matching (parameter count is low)
        self.proj_bn_to_L4 = nn.Conv2d(C_BN, C * 8, kernel_size=1) 
        self.proj_bn_to_L3 = nn.Conv2d(C_BN, C * 4, kernel_size=1) 

        # L2 Prediction Head (DELTA PREDICTION)
        self.pred_feat2_fused = nn.Sequential(
            ResidualBlock(C_L2_FUSION_IN, C * 2),
            nn.Conv2d(C * 2, C, kernel_size=1),  
            nn.GELU()
        )

        # L1 Prediction Head (DELTA PREDICTION)
        self.pred_feat1_fused = nn.Sequential(
            ResidualBlock(C_L1_FUSION_IN, C),
            nn.Conv2d(C, C // 2, kernel_size=1), 
            nn.GELU()
        )

        self.apply(init_weights)

    def forward(self, feats1u, feats2u, feats3u, feats4u, bottleneck_4d):
        N = bottleneck_4d.size(0)

        # --- 1. Evolve Bottleneck (Temporal State) ---
        E_bn = self.pred_bottleneck(bottleneck_4d)  
        E_bn = E_bn + bottleneck_4d 
        
        # --- LOW-COST UP-SAMPLING & PROJECTION ---
        # L4 Up-projection: 256ch -> 128ch (1x1 conv)
        E_bn_L4_proj = self.proj_bn_to_L4(E_bn)
        # Parameter-free Up-sampling (Interpolation)
        E_bn_L4_up = F.interpolate(E_bn_L4_proj, size=feats4u.shape[-2:], mode='nearest')

        # L3 Up-projection: 256ch -> 64ch (1x1 conv)
        E_bn_L3_proj = self.proj_bn_to_L3(E_bn)
        # Parameter-free Up-sampling (Interpolation)
        E_bn_L3_up = F.interpolate(E_bn_L3_proj, size=feats3u.shape[-2:], mode='nearest')
        
        # --- 2. Evolve Level 4 (Attention Delta with BN Fusion) ---
        tokens4 = feats4u.flatten(2).transpose(1, 2)
        attn_delta4 = self.pred_feat4_attn(tokens4)  
        attn_delta4_4d = attn_delta4.transpose(1, 2).reshape(N, self.te1_model_dim, 32, 32)
        
        # FUSION: Add evolved BN signal to the Attention Delta
        delta4_fused = attn_delta4_4d + E_bn_L4_up 
        
        # Evolved L4 = Input M4 + (CNN refinement of Fused Delta)
        # CRITICAL SKIP CHANGE: CNN output acts as the full delta/update to M4
        feats4u_evolved = feats4u + self.pred_feat4_cnn(delta4_fused)

        # --- 3. Evolve Level 3 (Attention Delta with BN Fusion) ---
        tokens3 = feats3u.flatten(2).transpose(1, 2)
        attn_delta3 = self.pred_feat3_attn(tokens3)
        attn_delta3_4d = attn_delta3.transpose(1, 2).reshape(N, self.te2_model_dim, 64, 64)
        
        # FUSION: Add evolved BN signal to the Attention Delta
        delta3_fused = attn_delta3_4d + E_bn_L3_up
        
        # Evolved L3 = Input M3 + (CNN refinement of Fused Delta)
        # CRITICAL SKIP CHANGE: CNN output acts as the full delta/update to M3
        feats3u_evolved = feats3u + self.pred_feat3_cnn(delta3_fused)
        
        # --- 4. Evolve Level 2 (Fused Path - DELTA PREDICTION) ---
        E_bn_L2_up = self.upsample_bn_to_L2(E_bn)  
        fused_L2 = torch.cat([feats2u, E_bn_L2_up], dim=1)  
        delta_L2 = self.pred_feat2_fused(fused_L2)
        feats2u_evolved = feats2u + delta_L2

        # --- 5. Evolve Level 1 (Fused Path - DELTA PREDICTION) ---
        E_bn_L1_up = self.upsample_bn_to_L1(E_bn)  
        fused_L1 = torch.cat([feats1u, E_bn_L1_up], dim=1)  
        delta_L1 = self.pred_feat1_fused(fused_L1)  
        feats1u_evolved = feats1u + delta_L1 
        
        return feats1u_evolved, feats2u_evolved, feats3u_evolved, feats4u_evolved, E_bn
   
    
# class DynNet(nn.Module):
#     """
#     Dynamics Network: Isolates feature estimation (M_t) from predictive state 
#     evolution (E_t+1). All deep layers (L5, L4, L3) use Attention Delta Prediction.
#     """
#     def __init__(self, args, base_channels=None):
#         super(DynNet, self).__init__()
        
#         C = base_channels if base_channels is not None else 16

#         self.te1_model_dim = C * 8 # L4: 128
#         self.te2_model_dim = C * 4 # L3: 64
#         self.te3_model_dim = C * 16 # L5: 256 - NEW DIMENSION

#         C_BN = C * 16
#         C_L2_FUSION_IN = C + (C_BN // 8)
#         C_L1_FUSION_IN = (C // 2) + (C_BN // 16)

#         # --- L5 Bottleneck Evolution (Attention Delta) ---
#         self.pred_bottleneck_attn = RoPETransformerEncoderLayer(
#             d_model=self.te3_model_dim, 
#             nhead=args.nhead, 
#             dim_feedforward=args.d_attn2, 
#             dropout=0, 
#             batch_first=True
#         )
#         self.pred_bottleneck_cnn = ResidualBlock(C * 16, C * 16)

#         # L4 Prediction Head (Attention Delta) - No change
#         self.pred_feat4_attn = RoPETransformerEncoderLayer(d_model=self.te1_model_dim, nhead=args.nhead, dim_feedforward=args.d_attn2, dropout=0, batch_first=True)
#         self.pred_feat4_cnn = ResidualBlock(C * 8, C * 8)
        
#         # L3 Prediction Head (Attention Delta) - No change
#         self.pred_feat3_attn = RoPETransformerEncoderLayer(d_model=self.te2_model_dim, nhead=args.nhead, dim_feedforward=args.d_attn1, dropout=0, batch_first=True)
#         self.pred_feat3_cnn = ResidualBlock(C * 4, C * 4)

#         # ... (Upsampling and Fused Path heads L1/L2 remain unchanged)
#         self.upsample_bn_to_L2 = nn.ConvTranspose2d(C_BN, C_BN // 8, kernel_size=8, stride=8)
#         self.upsample_bn_to_L1 = nn.ConvTranspose2d(C_BN, C_BN // 16, kernel_size=16, stride=16)

#         self.proj_bn_to_L4 = nn.Conv2d(C_BN, C * 8, kernel_size=1)
#         self.proj_bn_to_L3 = nn.Conv2d(C_BN, C * 4, kernel_size=1)

#         self.pred_feat2_fused = nn.Sequential(
#             ResidualBlock(C_L2_FUSION_IN, C * 2),
#             nn.Conv2d(C * 2, C, kernel_size=1),
#             nn.GELU()
#         )

#         self.pred_feat1_fused = nn.Sequential(
#             ResidualBlock(C_L1_FUSION_IN, C),
#             nn.Conv2d(C, C // 2, kernel_size=1),
#             nn.GELU()
#         )

#         self.apply(init_weights)

#     def forward(self, feats1u, feats2u, feats3u, feats4u, bottleneck_4d):
#         N = bottleneck_4d.size(0)
        
#         # --- 1. Evolve Bottleneck (L5) - ATTENTION DELTA (NEW) ---
#         tokens_bn = bottleneck_4d.flatten(2).transpose(1, 2)
#         attn_delta_bn = self.pred_bottleneck_attn(tokens_bn) # Attention Delta
#         attn_delta_bn_4d = attn_delta_bn.transpose(1, 2).reshape(N, self.te3_model_dim, 16, 16)
        
#         # Evolved L5 = Input M5 + (CNN refinement of Attention Delta)
#         E_bn = bottleneck_4d + self.pred_bottleneck_cnn(attn_delta_bn_4d)
        
#         # --- LOW-COST UP-SAMPLING & PROJECTION (Using E_bn) ---
#         E_bn_L4_proj = self.proj_bn_to_L4(E_bn)
#         E_bn_L4_up = F.interpolate(E_bn_L4_proj, size=feats4u.shape[-2:], mode='nearest')

#         E_bn_L3_proj = self.proj_bn_to_L3(E_bn)
#         E_bn_L3_up = F.interpolate(E_bn_L3_proj, size=feats3u.shape[-2:], mode='nearest')
        
#         # --- 2. Evolve Level 4 (L4) - ATTENTION DELTA (Unchanged logic) ---
#         tokens4 = feats4u.flatten(2).transpose(1, 2)
#         attn_delta4 = self.pred_feat4_attn(tokens4)
#         attn_delta4_4d = attn_delta4.transpose(1, 2).reshape(N, self.te1_model_dim, 32, 32)
#         delta4_fused = attn_delta4_4d + E_bn_L4_up
#         feats4u_evolved = feats4u + self.pred_feat4_cnn(delta4_fused)

#         # --- 3. Evolve Level 3 (L3) - ATTENTION DELTA (Unchanged logic) ---
#         tokens3 = feats3u.flatten(2).transpose(1, 2)
#         attn_delta3 = self.pred_feat3_attn(tokens3)
#         attn_delta3_4d = attn_delta3.transpose(1, 2).reshape(N, self.te2_model_dim, 64, 64)
#         delta3_fused = attn_delta3_4d + E_bn_L3_up
#         feats3u_evolved = feats3u + self.pred_feat3_cnn(delta3_fused)
        
#         # --- 4. Evolve Level 2 (L2) - FUSED DELTA (Unchanged logic) ---
#         E_bn_L2_up = self.upsample_bn_to_L2(E_bn)
#         fused_L2 = torch.cat([feats2u, E_bn_L2_up], dim=1)
#         delta_L2 = self.pred_feat2_fused(fused_L2)
#         feats2u_evolved = feats2u + delta_L2

#         # --- 5. Evolve Level 1 (L1) - FUSED DELTA (Unchanged logic) ---
#         E_bn_L1_up = self.upsample_bn_to_L1(E_bn)
#         fused_L1 = torch.cat([feats1u, E_bn_L1_up], dim=1)
#         delta_L1 = self.pred_feat1_fused(fused_L1)
#         feats1u_evolved = feats1u + delta_L1
        
#         return feats1u_evolved, feats2u_evolved, feats3u_evolved, feats4u_evolved, E_bn
    
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
    
# class UPredNet(nn.Module):
#     """
#     Simple Video Prediction system: 
#     Unet_Enc -> DynNet (Temporal Prediction) -> Unet_Dec
    
#     Optimized: Reuses the feature maps of the anchor frame (I0) for both 
#     Reconstruction (I0_hat) and Prediction paths, making the forward pass
#     efficient by calling the Encoder (E1) only once on the I0-I2 sequence.
#     """
#     # Changed img_channels and base_channels defaults to None to resolve NameError on module load
#     def __init__(self, args, img_channels=None, base_channels=None):
#         super(UPredNet, self).__init__()
        
#         # Resolve defaults inside __init__
#         if img_channels is None:
#             img_channels = args.img_channels
#         if base_channels is None:
#             base_channels = 16 # Assuming BASE_CHANNELS constant value is 16
            
#         # E1: Feature Extractor (Time t)
#         self.E1 = Unet_Enc(args, img_channels, base_channels)
#         # P: Temporal Feature Predictor (t -> t+1)
#         self.P = DynNet(args, base_channels)
#         # D1: Frame Reconstructor (Time t+1)
#         self.D1 = Unet_Dec(args, img_channels, base_channels)
        
#     def forward(self, input_clips):
#         """
#         Processes a clip (B, C, T, H, W) by performing two tasks efficiently:
#         1. Reconstructs I0 to I0_hat (Reusing E1 features).
#         2. Uses frames [0, T-2] to predict frames [1, T-1].
        
#         Returns: predictions, target_frames, I0_hat, I0_gt
#         """
#         B, C, T, H, W = input_clips.shape
#         T_pred = T - 1 # 3 predicted frames
        
#         # --------------------------------------------------------
#         # A. PREDICTION SEQUENCE SETUP (Input: I0, I1, I2)
#         # --------------------------------------------------------
        
#         # Ground Truth for Anchor (I0) - shape [B, C, H, W]
#         I0_gt = input_clips[:, :, 0, :, :] 

#         # Input for E1 (I0, I1, I2) - shape [B * (T-1), C, H, W]
#         input_frames_E1 = input_clips[:, :, :T-1, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
#         # Target for Prediction (I1, I2, I3) - shape [B * (T-1), C, H, W]
#         target_frames_pred = input_clips[:, :, 1:, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

#         # --------------------------------------------------------
#         # B. EFFICIENT FEATURE EXTRACTION (Single E1 call for all 3 frames)
#         # --------------------------------------------------------
#         # E1_features is a tuple of (feats1u, feats2u, feats3u, feats4u, bottleneck_4d)
#         E1_features = self.E1(input_frames_E1)
        
#         # The first B entries in each feature tensor correspond to I0
#         I0_E1_features = [f[:B] for f in E1_features]
#         # The rest of the features (B*(T-1) total) are used for prediction
        
#         # --------------------------------------------------------
#         # C. ANCHOR FRAME RECONSTRUCTION (I_hat_0) - REUSES I0_E1_features
#         # --------------------------------------------------------
#         I0_hat = self.D1(*I0_E1_features)
        
#         # --------------------------------------------------------
#         # D. TEMPORAL PREDICTION (DynNet + D1)
#         # --------------------------------------------------------
        
#         # Temporal Prediction (DynNet) uses all B*(T-1) features
#         feats1u_pred, feats2u_pred, feats3u_pred, feats4u_pred, bottleneck_4d_pred = \
#             self.P(*E1_features) # Use all features extracted in step B

#         # Frame Reconstruction (D1) for predicted features
#         out_frames_pred = self.D1(feats1u_pred, feats2u_pred, feats3u_pred, feats4u_pred, bottleneck_4d_pred)
        
#         # Reshape prediction output back to clip format: [B, C, T-1, H, W]
#         predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        
#         # Return predicted sequence, its ground truth, reconstructed anchor, and anchor ground truth
#         return predictions, target_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4), I0_hat, I0_gt

# class UPredNet(nn.Module):
#     """
#     UPredNet Baseline using a feed-forward DynNet, augmented with Channel Fusion 
#     Blocks (CFB) before and after dynamics for fair comparison with SWAU_Net.
    
#     This serves as the "no SWA" baseline to test the efficacy of the SWA module.
#     """
#     def __init__(self, args, img_channels=None, base_channels=None):
#         super().__init__()
        
#         # Safely resolve channel values inside the function body
#         base_channels = base_channels if base_channels is not None else BASE_CHANNELS
#         img_channels = img_channels if img_channels is not None else 3 # Default to 3 for standard color/FAF
        
#         # E1: Feature Extractor (Time t)
#         self.E1 = Unet_Enc(args, img_channels, base_channels)
        
#         # CFB_enc: Pre-Dynamics Feature Refinement (New Addition)
#         self.CFB_enc = ChannelFusionBlock(base_channels) 
        
#         # P: Temporal Feature Predictor (DynNet)
#         # Assuming P is the SIMPLIFIED DynNet (Conv-only) for fair comparison
#         self.P = DynNet(args, base_channels)
        
#         # CFB_dec: Post-Dynamics Feature Refinement (New Addition)
#         self.CFB_dec = ChannelFusionBlock(base_channels)
        
#         # D1: Frame Reconstructor (Time t+1)
#         self.D1 = Unet_Dec(args, img_channels, base_channels)

#     def forward(self, input_clips):
#         """
#         Processes a clip (B, C, T, H, W).
#         Flow: E1 -> CFB_enc -> DynNet -> CFB_dec -> D1
#         """
#         B, C, T, H, W = input_clips.shape
#         T_pred = T - 1 # e.g., 3 predicted frames
        
#         # --- A. ENCODER INPUT PREPARATION ---
#         I0_gt = input_clips[:, :, 0, :, :] 
#         # Input for E1 (I0, I1, I2)
#         input_frames_E1 = input_clips[:, :, :T_pred, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
#         # --- B. EFFICIENT FEATURE EXTRACTION (E1) ---
#         E1_features_flat = self.E1(input_frames_E1)

#         # --- C. PRE-DYNAMICS CHANNEL FUSION (CFB_enc) ---
#         # F' = CFB_enc(F). Refined features before DynNet (M_t = F'_t since no SWA)
#         F_refined_flat = self.CFB_enc(E1_features_flat)
        
#         # --- D. ANCHOR FRAME RECONSTRUCTION (I_hat_0) ---
#         # Features for I0 use the first B refined features
#         I0_F_refined = [f[:B] for f in F_refined_flat]
#         I0_hat = self.D1(*I0_F_refined)
        
#         # --- E. TEMPORAL PREDICTION (DynNet) ---
#         # E' = P(F'). DynNet treats the refined features F' as its input M.
#         E_raw_evolved_flat = self.P(*F_refined_flat)
        
#         # --- F. POST-DYNAMICS CHANNEL FUSION (CFB_dec) ---
#         # E = CFB_dec(E'). Polished features E for decoding.
#         Evolved_polished = self.CFB_dec(E_raw_evolved_flat)
        
#         E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = Evolved_polished

#         # --- G. DECODING & RESHAPING ---
#         out_frames_pred = self.D1(*Evolved_polished)
        
#         # Reshape prediction output back to clip format: [B, C, T-1, H, W]
#         predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        
#         # Targets are correctly sliced to (N, C, T_pred, H, W)
#         targets = input_clips[:, :, 1:, :, :]
        
#         # Target ground truth is the prediction slice.
#         target_frames_pred = targets.reshape(-1, C, H, W)

#         # Return 9 items for consistent loss tracking across all models
#         return predictions, targets, I0_hat, I0_gt, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved

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

# --- CausalConvAggregator Helper (Based on SWA's CNN Mixers) ---

class CausalConvAggregator(nn.Module):
    """
    Replaces SWA by implementing only the CNN mixing/aggregation logic, 
    running it over the 3 required causal windows (M0, M1, M2).
    This outputs M_flat ready for DynNet.
    """
    def __init__(self, args, base_channels=16):
        super().__init__()
        C = base_channels
        
        # Replicate SWA's CNN heads for L1, L2, L3, but NO ATTENTION layers.
        # These are simple CNNs designed to synthesize 3 input frames into 1 output frame (M_t).
        
        # L3 Mixer (LocalSpatioTemporalMixer logic)
        self.mixer_l3 = LocalSpatioTemporalMixer(in_channels=C * 4, kernel_size=3) 
        
        # L2 Conv Head: Input 3C -> Output C
        self.conv_l2 = nn.Sequential(
            nn.Conv2d(C * 3, C, kernel_size=1), 
            nn.GELU(),
            ResidualBlock(C, C) 
        )
        
        # L1 Conv Head: Input 1.5C -> Output C/2
        self.conv_l1 = nn.Sequential(
            nn.Conv2d(C * 3 // 2, C // 2, kernel_size=1),
            nn.GELU(),
            ResidualBlock(C // 2, C // 2)
        )
        
        # Bottleneck (L5) and L4 use direct CNN concatenation and reduction
        # L5 (BN): Input 16C * 3 -> Output 16C
        C_BN = C * 16
        self.conv_bn = nn.Sequential(
            nn.Conv2d(C_BN * 3, C_BN, kernel_size=1),
            nn.GELU(),
            ResidualBlock(C_BN, C_BN)
        )
        
        # L4: Input 8C * 3 -> Output 8C
        C_L4 = C * 8
        self.conv_l4 = nn.Sequential(
            nn.Conv2d(C_L4 * 3, C_L4, kernel_size=1),
            nn.GELU(),
            ResidualBlock(C_L4, C_L4)
        )
        
        self.apply(init_weights)

    def forward(self, F_t_minus_2, F_t_minus_1, F_t):
        """
        Takes features for one 3-frame window and outputs a single aggregated state M_t.
        """
        f1_t_2, f2_t_2, f3_t_2, f4_t_2, bn_t_2 = F_t_minus_2
        f1_t_1, f2_t_1, f3_t_1, f4_t_1, bn_t_1 = F_t_minus_1
        f1_t, f2_t, f3_t, f4_t, bn_t = F_t

        # 1. L5 (BN) - Aggregation
        bn_cat = torch.cat([bn_t_2, bn_t_1, bn_t], dim=1)
        m_bn = self.conv_bn(bn_cat)
        m_bn = m_bn + bn_t # Macro-residual skip

        # 2. L4 - Aggregation
        f4_cat = torch.cat([f4_t_2, f4_t_1, f4_t], dim=1)
        m4 = self.conv_l4(f4_cat)
        m4 = m4 + f4_t

        # 3. L3 - Aggregation (Uses the LocalSpatioTemporalMixer logic)
        m3 = self.mixer_l3(f3_t_2, f3_t_1, f3_t) # m3 already includes residual to f3_t

        # 4. L2 - Aggregation
        f2_cat = torch.cat([f2_t_2, f2_t_1, f2_t], dim=1)
        m2 = self.conv_l2(f2_cat)
        m2 = m2 + f2_t

        # 5. L1 - Aggregation
        f1_cat = torch.cat([f1_t_2, f1_t_1, f1_t], dim=1)
        m1 = self.conv_l1(f1_cat)
        m1 = m1 + f1_t
        
        # Returns 5 tensors, each representing the single aggregated state M_t
        return m1, m2, m3, m4, m_bn
    
    
# --- Updated UPredNet with Causal CNN Aggregation ---

class UPredNet(nn.Module):
    """
    UPredNet Baseline using a strict Causal Convolutional Aggregator 
    (CausalConvAggregator) in a sliding window to generate single-frame states M_t.
    This provides a clean, Causal, Non-Attention Baseline.
    """
    def __init__(self, args, img_channels=None, base_channels=None):
        super().__init__()
        
        base_channels = base_channels if base_channels is not None else 16
        img_channels = img_channels if img_channels is not None else 3
        
        # E1, CFB_enc, CFB_dec, D1, P (DynNet) remain the same
        self.E1 = Unet_Enc(args, img_channels, base_channels)
        self.CFB_enc = ChannelFusionBlock(base_channels) 
        self.CFB_dec = ChannelFusionBlock(base_channels)
        self.P = DynNet(args, base_channels)
        self.D1 = Unet_Dec(args, img_channels, base_channels)

        # *** KEY CHANGE: The new Causal Aggregator replaces the SWA/simple pass ***
        self.Causal_Aggregator = CausalConvAggregator(args, base_channels)
        
    def forward(self, input_clips):
        B, C, T, H, W = input_clips.shape
        T_pred = T - 1 # 3 predicted frames (I1, I2, I3)
        
        # --- A. FEATURE EXTRACTION & REFINE (Parallel Batch) ---
        I0_gt = input_clips[:, :, 0, :, :] 
        input_frames_E1 = input_clips[:, :, :T_pred, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        E1_features_flat = self.E1(input_frames_E1)
        
        # F' = Refined Features [F'0, F'1, F'2]
        F_refined_flat = self.CFB_enc(E1_features_flat)
        
        # Unpack F' features by time step 
        F0_fused = [f[:B] for f in F_refined_flat] 
        F1_fused = [f[B:2*B] for f in F_refined_flat] 
        F2_fused = [f[2*B:3*B] for f in F_refined_flat] 
        
        # --- B. ANCHOR RECONSTRUCTION ---
        I0_hat = self.D1(*F0_fused)

        # --- C. CAUSAL AGGREGATION (Windowing Loop) ---
        # Initialize zero feature list for padding (T-2, T-1, T)
        Zero_fused = [torch.zeros_like(f) for f in F0_fused]
        
        M_list = []
        
        # Window 0: Predicts I1. Context: (0, 0, F'0)
        M0_features = self.Causal_Aggregator(Zero_fused, Zero_fused, F0_fused) 
        M_list.append(M0_features)
        
        # Window 1: Predicts I2. Context: (0, F'0, F'1)
        M1_features = self.Causal_Aggregator(Zero_fused, F0_fused, F1_fused) 
        M_list.append(M1_features)
        
        # Window 2: Predicts I3. Context: (F'0, F'1, F'2)
        M2_features = self.Causal_Aggregator(F0_fused, F1_fused, F2_fused) 
        M_list.append(M2_features)

        # Re-stack single-frame M features into one large batch for DynNet
        M_flat = [
            torch.cat([M0_features[i], M1_features[i], M2_features[i]], dim=0) 
            for i in range(5)
        ]
        
        # --- D. TEMPORAL EVOLUTION (DynNet) ---
        # DynNet processes the M_flat batch in parallel.
        Evolved_flat = self.P(*M_flat)
        
        # --- E. POST-DYNAMICS CHANNEL FUSION (CFB_dec) and Decoding ---
        Evolved_polished = self.CFB_dec(Evolved_flat)
        E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = Evolved_polished

        out_frames_pred = self.D1(*Evolved_polished)
        predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        targets = input_clips[:, :, 1:, :, :] 
        
        # Return 9 items
        return predictions, targets, I0_hat, I0_gt, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved
    
  