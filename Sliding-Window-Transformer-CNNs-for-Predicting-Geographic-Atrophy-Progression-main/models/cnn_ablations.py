import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import init_weights

from .autoencoder import ResidualBlock
from .CFB import ChannelFusionBlock
from .SWA import SlidingWindowAttention

# ------------------------------------------------------------------------------
# 1. Core CNN Blocks (AGRB = Adaptive Gated Residual Block)
# ------------------------------------------------------------------------------

class AdaptiveGatedResidualBlock(nn.Module):
    """
    Residual Block with Gated Fusion (Original ResidualBlock implementation).
    Used to maintain the same convolutional expressivity and parameter count
    as the original architecture.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        
        # 1. Main Convolutional Path (f(x))
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        
        # 2. Detail Residual Path (High-Frequency)
        self.detail_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        
        # Trainable Scalar Gate for Detail Path. Initialized small (0.1)
        self.detail_weight = nn.Parameter(torch.ones(1) * 0.1) 

        # 3. Shortcut Connection (Identity or 1x1 Conv for dimension matching)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

        self.gelu = nn.GELU()
        # self.apply(init_weights) # Assuming init_weights is defined elsewhere

    def forward(self, x):
        main_out = self.conv(x)
        detail_out = self.detail_conv(x)
        shortcut_out = self.shortcut(x)
        
        # Gated Fusion: Detail path output is scaled by the learned weight
        out = main_out + (self.detail_weight * detail_out) + shortcut_out
        
        out = self.gelu(out)
        return out

class ChannelReducer(nn.Module):
    """
    Reduces the number of channels using a simple 1x1 convolution.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        # self.apply(init_weights) 
        
    def forward(self, x):
        return self.conv(x)

# ------------------------------------------------------------------------------
# 2. Ablated U-Net Encoder (CNN_Unet_Enc)
# ------------------------------------------------------------------------------

class CNN_Unet_Enc(nn.Module):
    """
    U-Net Encoder with SPATIAL ATTENTION REMOVED.
    The RoPE Transformer Encoder Layer (self.te_sa4_1) is replaced by a simple
    AdaptiveGatedResidualBlock to maintain parameter count and convolutional feature
    mixing without non-local self-attention.
    """
    def __init__(self, args, img_channels=3, base_channels=16):
        super(CNN_Unet_Enc, self).__init__()
        C = base_channels 
        
        # 1. FAF Pathway (Input: 1 channel, Output: C/4 channels)
        self.conv1_faf = AdaptiveGatedResidualBlock(1, C // 4) 
        
        # 2. Geometric Pathway (Input: 2 channels, Output: 3C/4 channels)
        self.conv1_geo = AdaptiveGatedResidualBlock(2, C * 3 // 4) 
        
        # --- Base U-Net Path ---
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = AdaptiveGatedResidualBlock(C, C * 2) 
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = AdaptiveGatedResidualBlock(C * 2, C * 4)
        self.dropout3 = nn.Dropout2d(0.2) 
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = AdaptiveGatedResidualBlock(C * 4, C * 8)
        self.dropout4 = nn.Dropout2d(0.2) 
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # **ABLATION: Replace Spatial Attention with a CNN block**
        # This CNN block maintains channel dimensions (C*8 -> C*8) 
        # and has similar complexity but lacks non-local interaction.
        self.cnn_attn_replacement = AdaptiveGatedResidualBlock(C * 8, C * 8)
        
        self.conv5 = AdaptiveGatedResidualBlock(C * 8, C * 16)

        # Skip Channel Reducers (Used in skip connections and for DynNet input)
        self.reduce_l1 = ChannelReducer(C, C // 2) 
        self.reduce_l2 = ChannelReducer(C * 2, C) 

        # self.apply(init_weights)

    def forward(self, in_frames):
        # N is total batch size (B * T_pred)
        N = in_frames.size(0)
        
        # Split input: (FAF, Mask+Residual)
        faf_input = in_frames[:, 0:1, :, :] 
        geo_input = in_frames[:, 1:3, :, :] 

        # L1 (256x256) - Dual Path Encoding
        feats_faf = self.conv1_faf(faf_input)       # Output C/4 channels
        feats_geo = self.conv1_geo(geo_input)       # Output 3C/4 channels
        feats1u_full = torch.cat((feats_faf, feats_geo), dim=1) # Total C channels
        
        feats1u_reduced = self.reduce_l1(feats1u_full)
        x = self.pool1(feats1u_full)
        
        # L2 (128x128) and onward
        feats2u_full = self.conv2(x)
        feats2u_reduced = self.reduce_l2(feats2u_full)
        x = self.pool2(feats2u_full)
        
        feats3u = self.conv3(x)
        feats3u = self.dropout3(feats3u)
        x_pre_attn = self.pool3(feats3u) 
        feats4u_pre = self.conv4(x_pre_attn)
        feats4u_pre = self.dropout4(feats4u_pre)

        # **ABLATION: Apply CNN Replacement (instead of tokenization/attention)**
        feats4u = self.cnn_attn_replacement(feats4u_pre)
        
        x_bottleneck_pre = self.pool4(feats4u)
        bottleneck_4d = self.conv5(x_bottleneck_pre) 
        
        # Return skip features and bottleneck output
        return feats1u_reduced, feats2u_reduced, feats3u, feats4u, bottleneck_4d
        
# ------------------------------------------------------------------------------
# 3. Ablated U-Net Decoder (CNN_Unet_Dec)
# ------------------------------------------------------------------------------

class CNN_Unet_Dec(nn.Module):
    """
    U-Net Decoder with UNUSED ATTENTION COMPONENTS REMOVED from __init__.
    The forward path remains identical to the original Unet_Dec as it did not
    use the decoder-side attention layers.
    """
    def __init__(self, args, img_channels=3, base_channels=16):
        super(CNN_Unet_Dec, self).__init__()
        C = base_channels 

        # --- L4 Setup (C*16 -> C*8) ---
        self.upconv4 = nn.ConvTranspose2d(C * 16, C * 8, kernel_size=2, stride=2)
        # Input channel count: Upconv output (C*8) + Skip connection (C*8) = C*16
        self.up_conv4 = AdaptiveGatedResidualBlock(C * 8 + C * 8, C * 8)
        
        # ATTENTION LAYERS REMOVED: self.te_sa4d_1 and self.mha4_ca are omitted

        # --- L3 Setup (C*8 -> C*4) ---
        self.upconv3 = nn.ConvTranspose2d(C * 8, C * 4, kernel_size=2, stride=2)
        self.up_conv3 = AdaptiveGatedResidualBlock(C * 4 + C * 4, C * 4)

        # ATTENTION LAYERS REMOVED: self.te_sa3d_1 and self.mha3_ca are omitted

        # --- L2 Setup (ADDITIVE Skip Connection) ---
        self.upconv2 = nn.ConvTranspose2d(C * 4, C * 2, kernel_size=2, stride=2) 
        self.l2_add_match = nn.Conv2d(C, C * 2, kernel_size=1) 
        self.up_conv2 = AdaptiveGatedResidualBlock(C * 2, C * 2) 

        # --- L1 Setup (ADDITIVE Skip Connection) ---
        self.upconv1 = nn.ConvTranspose2d(C * 2, C, kernel_size=2, stride=2) 
        self.l1_add_match = nn.Conv2d(C // 2, C, kernel_size=1) 
        self.up_conv1 = AdaptiveGatedResidualBlock(C, C) 
        
        # Final output layer
        self.final_conv = nn.Conv2d(C, img_channels, kernel_size=1)
        self.sig = nn.Sigmoid()

        # self.apply(init_weights) 
        
    def forward(self, feats1u_enc, feats2u_enc, feats3u_enc, feats4u_enc, bottleneck_4d):
        
        # Level 4: Upsample and Concatenate Skip
        x = self.upconv4(bottleneck_4d)
        x = torch.cat([x, feats4u_enc], dim=1) 
        feats4d = self.up_conv4(x)

        # Level 3: Upsample and Concatenate Skip
        x = self.upconv3(feats4d)
        x = torch.cat([x, feats3u_enc], dim=1) 
        feats3d = self.up_conv3(x)

        # Level 2: Upsample and ADDITIVE Skip
        x = self.upconv2(feats3d)
        skip_matched = self.l2_add_match(feats2u_enc)
        x = x + skip_matched
        feats2d = self.up_conv2(x)

        # Level 1: Upsample and ADDITIVE Skip
        x = self.upconv1(feats2d)
        skip_matched = self.l1_add_match(feats1u_enc)
        x = x + skip_matched
        feats1d = self.up_conv1(x)
        
        # Final output
        out_frames_logits = self.final_conv(feats1d)
        out_frames = self.sig(out_frames_logits)

        return out_frames

# ------------------------------------------------------------------------------
# 4. Ablated U-Net Wrapper (CNN_U_Net_AE)
# ------------------------------------------------------------------------------

class CNN_U_Net_AE(nn.Module):
    """ 
    Wrapper combining the CNN-only U-Net Encoder and the CNN-only Decoder
    for ablation studies.
    """
    def __init__(self, args, img_channels=3, base_channels=16): 
        super(CNN_U_Net_AE, self).__init__()
        # Use the ablated, CNN-only components
        self.E1 = CNN_Unet_Enc(args, img_channels, base_channels)
        self.D1 = CNN_Unet_Dec(args, img_channels, base_channels)
        
    def forward(self, in_frames):
        """
        Performs the full auto-encoding process.
        """
        feats1u, feats2u, feats3u, feats4u, bottleneck_4d = self.E1(in_frames)
        out_frames = self.D1(feats1u, feats2u, feats3u, feats4u, bottleneck_4d)
        
        # Return reconstruction and all skip features
        return out_frames, feats1u, feats2u, feats3u, feats4u, bottleneck_4d
    
    
    
class CNN_DynNet(nn.Module):
    """
    CNN-only version of DynNet with all Transformer components REMOVED.
    The L4 and L3 paths use CNN fusion to predict the feature delta (Delta_x).
    """
    def __init__(self, args, base_channels=None):
        super(CNN_DynNet, self).__init__()
        
        C = base_channels if base_channels is not None else 16

        C_BN = C * 16 # 256
        
        # --- Channel Calculation for Fused Inputs (Based on Encoder Channel Counts) ---
        
        # L4 Input: M4 (C*8) + E_bn_L4_up (C*8) = C*16 = 256
        C_L4_FUSION_IN = C * 16 
        
        # L3 Input: M3 (C*4) + E_bn_L3_up (C*4) = C*8 = 128
        C_L3_FUSION_IN = C * 8 
        
        # L2 Input: M2 (C) + E_bn_L2_up (C*2) = C*3 = 48
        C_L2_FUSION_IN = C * 3
        
        # L1 Input: M1 (C/2) + E_bn_L1_up (C) = C*1.5 = 24
        C_L1_FUSION_IN = C * 3 // 2 

        # --- Core Evolution Heads (CNN-ONLY) ---
        self.pred_bottleneck = ResidualBlock(C * 16, C * 16)
        
        # **FIXED L4: Input 256 channels (16C) -> Output Delta 8C (128 channels)**
        self.pred_feat4_evolve = nn.Sequential(
            ResidualBlock(C_L4_FUSION_IN, C * 8), # Maps 256 -> 128
            ResidualBlock(C * 8, C * 8), # Refinement block
        )
        
        # **FIXED L3: Input 128 channels (8C) -> Output Delta 4C (64 channels)**
        self.pred_feat3_evolve = nn.Sequential(
            ResidualBlock(C_L3_FUSION_IN, C * 4), # Maps 128 -> 64
            ResidualBlock(C * 4, C * 4), # Refinement block
        )

        # --- Upsampling/Projection Layers ---
        # NOTE: Transposed Convolutions are still used to UP-project the BN state
        self.upsample_bn_to_L2 = nn.ConvTranspose2d(C_BN, C_BN // 8, kernel_size=8, stride=8)  # 256 -> 32 channels (C*2)
        self.upsample_bn_to_L1 = nn.ConvTranspose2d(C_BN, C_BN // 16, kernel_size=16, stride=16) # 256 -> 16 channels (C)

        # NOTE: These projections are now used to match BN channels to the input size before interpolation
        self.proj_bn_to_L4 = nn.Conv2d(C_BN, C * 8, kernel_size=1) 
        self.proj_bn_to_L3 = nn.Conv2d(C_BN, C * 4, kernel_size=1) 

        # L2 Prediction Head (Fused-Path Delta Prediction - CNN-only)
        self.pred_feat2_fused = nn.Sequential(
            ResidualBlock(C_L2_FUSION_IN, C * 2),
            nn.Conv2d(C * 2, C, kernel_size=1),  # Output Delta (C)
            nn.GELU()
        )

        # L1 Prediction Head (Fused-Path Delta Prediction - CNN-only)
        self.pred_feat1_fused = nn.Sequential(
            ResidualBlock(C_L1_FUSION_IN, C),
            nn.Conv2d(C, C // 2, kernel_size=1), # Output Delta (C/2)
            nn.GELU()
        )

        # self.apply(init_weights)

    def forward(self, feats1u, feats2u, feats3u, feats4u, bottleneck_4d):
        N = bottleneck_4d.size(0)

        # --- 1. Evolve Bottleneck (Temporal State) ---
        E_bn = self.pred_bottleneck(bottleneck_4d)  
        E_bn = E_bn + bottleneck_4d 
        
        # --- LOW-COST UP-SAMPLING & PROJECTION (E_bn -> L4/L3) ---
        # L4 Up-projection: C*16 -> C*8 (1x1 conv)
        E_bn_L4_proj = self.proj_bn_to_L4(E_bn)
        # L4 Up-sampling (Interpolation)
        E_bn_L4_up = F.interpolate(E_bn_L4_proj, size=feats4u.shape[-2:], mode='nearest')

        # L3 Up-projection: C*16 -> C*4 (1x1 conv)
        E_bn_L3_proj = self.proj_bn_to_L3(E_bn)
        # L3 Up-sampling (Interpolation)
        E_bn_L3_up = F.interpolate(E_bn_L3_proj, size=feats3u.shape[-2:], mode='nearest')
        
        # --- 2. Evolve Level 4 (CNN-ONLY Fusion Delta) ---
        # Concatenate M4 (128) and E_bn_L4_up (128) -> 256 channels
        fused_L4 = torch.cat([feats4u, E_bn_L4_up], dim=1) 
        # Predict Delta (Delta4) from 256 channels to 128 channels
        delta_L4 = self.pred_feat4_evolve(fused_L4)
        feats4u_evolved = feats4u + delta_L4

        # --- 3. Evolve Level 3 (CNN-ONLY Fusion Delta) ---
        # Concatenate M3 (64) and E_bn_L3_up (64) -> 128 channels
        fused_L3 = torch.cat([feats3u, E_bn_L3_up], dim=1) 
        # Predict Delta (Delta3) from 128 channels to 64 channels
        delta_L3 = self.pred_feat3_evolve(fused_L3)
        feats3u_evolved = feats3u + delta_L3
        
        # --- 4. Evolve Level 2 (CNN-ONLY Fused Path - DELTA PREDICTION) ---
        # Transposed conv upsamples E_bn -> 32 channels (C*2)
        E_bn_L2_up = self.upsample_bn_to_L2(E_bn)  
        # Concatenate M2 (16) and E_bn_L2_up (32) -> 48 channels
        fused_L2 = torch.cat([feats2u, E_bn_L2_up], dim=1)  
        delta_L2 = self.pred_feat2_fused(fused_L2)
        feats2u_evolved = feats2u + delta_L2

        # --- 5. Evolve Level 1 (CNN-ONLY Fused Path - DELTA PREDICTION) ---
        # Transposed conv upsamples E_bn -> 16 channels (C)
        E_bn_L1_up = self.upsample_bn_to_L1(E_bn)  
        # Concatenate M1 (8) and E_bn_L1_up (16) -> 24 channels
        fused_L1 = torch.cat([feats1u, E_bn_L1_up], dim=1)  
        delta_L1 = self.pred_feat1_fused(fused_L1)  
        feats1u_evolved = feats1u + delta_L1 
        
        return feats1u_evolved, feats2u_evolved, feats3u_evolved, feats4u_evolved, E_bn


class SWAU_Net_CNN(nn.Module):
    """
    SWAU_Net Ablation: Removes all explicit Spatial Self-Attention (Encoder and DynNet)
    by swapping the attention-based components for their CNN-only counterparts.
    
    Architecture: CNN_Enc -> CFB_enc -> SWA (Temporal Axial Attention, but CNN Mixers) 
                  -> CNN_DynNet -> CFB_dec -> CNN_Dec
                  
    This isolates the performance contribution of the SWA temporal regularization.
    """
    def __init__(self, args, img_channels=None, base_channels=None):
        super(SWAU_Net_CNN, self).__init__()
        
        # Resolve defaults inside __init__
        if img_channels is None:
            img_channels = 3 # Hardcoding initial default value
        if base_channels is None:
            base_channels = 16 # Hardcoding initial default value
            
        # 1. ENCODER: Use CNN-only Encoder (Spatial Attention removed)
        self.E1 = CNN_Unet_Enc(args, img_channels, base_channels)
        
        # 2. CFB_enc: Pre-Dynamics Fusion (Retained as per request)
        self.CFB_enc = ChannelFusionBlock(base_channels) 
        
        # 3. CFB_dec: Post-Dynamics Refinement (Retained as per request)
        self.CFB_dec = ChannelFusionBlock(base_channels) 
        
        # 4. SWA: Feature Aggregator (Retained as per request)
        self.SWA = SlidingWindowAttention(args, base_channels)
        
        # 5. DYN-NET: Use CNN-only DynNet (Attention Delta replaced by CNN Delta)
        self.P = CNN_DynNet(args, base_channels)
        
        # 6. DECODER: Use CNN-only Decoder (Attention components removed from __init__)
        self.D1 = CNN_Unet_Dec(args, img_channels, base_channels)
        
    def forward(self, input_clips):
        B, C, T, H, W = input_clips.shape
        T_pred = T - 1 # e.g., 3 predicted frames (I1, I2, I3)
        
        # --- A. FEATURE EXTRACTION & INPUT PREP ---
        I0_gt = input_clips[:, :, 0, :, :] 
        
        # Input to E1: Permute to (N, T, C, H, W) then reshape to (N*T, C, H, W)
        input_frames_E1 = input_clips[:, :, :T_pred, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
        # **E1 uses CNN_Unet_Enc (No spatial attention in encoder bottleneck)**
        E1_features_flat = self.E1(input_frames_E1)
        
        # --- 1. PRE-DYNAMICS CHANNEL FUSION (CFB_enc) ---
        FUSED_features_flat = self.CFB_enc(E1_features_flat)
        
        # Unpack FUSED features by time step 
        F0_fused = [f[:B] for f in FUSED_features_flat] 
        F1_fused = [f[B:2*B] for f in FUSED_features_flat] 
        F2_fused = [f[2*B:3*B] for f in FUSED_features_flat] 
        
        # --- B. ANCHOR RECONSTRUCTION (I_hat_0) ---
        # **D1 uses CNN_Unet_Dec**
        I0_hat = self.D1(*F0_fused)

        # --- C. SLIDING WINDOW ATTENTION (SWA) ---
        # **SWA block is retained, providing temporal attention**
        M0_features = self.SWA(F0_fused, F0_fused, F0_fused) 
        M1_features = self.SWA(F0_fused, F0_fused, F1_fused) 
        M2_features = self.SWA(F0_fused, F1_fused, F2_fused) 

        M_flat = [
            torch.cat([M0_features[i], M1_features[i], M2_features[i]], dim=0) 
            for i in range(5)
        ]
        
        # --- D. TEMPORAL EVOLUTION (CNN_DynNet) ---
        # **P uses CNN_DynNet (No attention in feature evolution)**
        Evolved_flat = self.P(*M_flat)
        
        # --- 2. POST-DYNAMICS CHANNEL FUSION (CFB_dec) ---
        Evolved_polished = self.CFB_dec(Evolved_flat)
        
        E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = Evolved_polished

        # --- E. DECODING & RESHAPING ---
        out_frames_pred = self.D1(*Evolved_polished)
        
        predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        targets = input_clips[:, :, 1:, :, :] 
        
        # Return 9 items
        return predictions, targets, I0_hat, I0_gt, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved