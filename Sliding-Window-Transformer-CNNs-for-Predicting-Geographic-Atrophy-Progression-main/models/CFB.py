import torch
import torch.nn as nn
import torch.nn.functional as F 

from .autoencoder import RoPEMultiheadAttention, RoPETransformerEncoderLayer
from .autoencoder import Unet_Enc, Unet_Dec, ResidualBlock
from .model_utils import init_weights

# ------------------------------------------------------------------------------
# CHANNEL FUSION BLOCK
# ------------------------------------------------------------------------------

class FusionBlockBottleneck(nn.Module):
    """
    A lightweight Bottleneck Block for Channel Fusion.
    Uses 1x1 Convs to reduce complexity around the 3x3 Conv, 
    retaining spatial interaction while minimizing parameters.
    """
    def __init__(self, channels, reduction_ratio=2):
        super().__init__()
        C_in = channels
        C_mid = C_in // reduction_ratio 
        
        self.conv_block = nn.Sequential(
            # 1. Bottleneck (Channel Reduction: C_in -> C_mid)
            nn.Conv2d(C_in, C_mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_mid),
            nn.GELU(),
            
            # 2. Core Spatial Mixing (3x3 at reduced channels)
            nn.Conv2d(C_mid, C_mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C_mid),
            nn.GELU(),
            
            # 3. Expansion (Channel Restoration: C_mid -> C_in)
            nn.Conv2d(C_mid, C_in, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_in)
        )
        
        # Additive skip connection is applied at the end
        self.gelu = nn.GELU()
        self.apply(init_weights)

    def forward(self, x):
        out = self.conv_block(x)
        out = x + out # Skip connection
        return self.gelu(out)


class ChannelFusionBlock(nn.Module):
    """
    Final ChannelFusionBlock using FusionBlockBottleneck.
    Initializes fusion blocks to match the REDUCED channel outputs 
    of Unet_Enc's skip connections (C/2 for L1, C for L2).
    """
    def __init__(self, base_channels):
        super().__init__()
        C = base_channels # Should be 16

        # L1: C // 2 = 8 channels
        self.fusion_l1 = FusionBlockBottleneck(C // 2) 
        
        # L2: C = 16 channels
        self.fusion_l2 = FusionBlockBottleneck(C) 
        
        # L3: 4C (64)
        self.fusion_l3 = FusionBlockBottleneck(C * 4)
        
        # L4: 8C (128)
        self.fusion_l4 = FusionBlockBottleneck(C * 8)
        
        # BN: 16C (256)
        self.fusion_bn = FusionBlockBottleneck(C * 16)
        
        self.apply(init_weights) 

    def forward(self, features):
        feats1u, feats2u, feats3u, feats4u, bottleneck_4d = features

        fused1 = self.fusion_l1(feats1u)
        fused2 = self.fusion_l2(feats2u)
        fused3 = self.fusion_l3(feats3u)
        fused4 = self.fusion_l4(feats4u)
        fused_bn = self.fusion_bn(bottleneck_4d)
        
        return fused1, fused2, fused3, fused4, fused_bn
  