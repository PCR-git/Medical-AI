import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import init_weights

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
    