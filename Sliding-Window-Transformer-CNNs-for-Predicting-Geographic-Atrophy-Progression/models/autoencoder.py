import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import init_weights

# --- UTILITIES: RoPE IMPLEMENTATION ---

def rotate_half(x):
    """
    Performs a rotation on the last dimension of the tensor by swapping 
    the two halves and negating the second half. This is a core operation 
    for Rotary Positional Embedding (RoPE).
    """
    D_H = x.shape[-1]
    # Reshape to separate halves: [..., D_H // 2, 2]
    x = x.reshape(*x.shape[:-1], D_H // 2, 2)  
    x1, x2 = x.unbind(dim=-1)
    # Perform rotation: [-x2, x1]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    return rotated_x

class RotaryPositionalEmbedding(nn.Module):
    """
    Generates the sinusoidal and cosinusoidal frequency vectors used for RoPE.
    The frequency vectors are computed based on the inverse frequency formulation.
    """
    def __init__(self, dim, max_seq_len=4096):
        """
        Args:
            dim (int): The feature dimension of a single attention head.
            max_seq_len (int): The maximum sequence length supported by the embedding.
        """
        super().__init__()
        # Calculate the inverse frequency vector
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, seq_len, device):
        """
        Generates the (cos, sin) rotation matrix for the given sequence length.

        Args:
            seq_len (int): The current sequence length (L).
            device (torch.device): The target device for the output tensor.
            
        Returns:
            torch.Tensor: Tensor of shape (2, L, D), where 2 holds (cos, sin).
        """
        # Time steps/sequence positions
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # Outer product to get frequencies: [L, D/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Duplicate frequencies to match the full dimension D: [L, D]
        emb = torch.cat((freqs, freqs), dim=-1)
        # Compute the sine and cosine components
        pe = torch.stack([emb.cos(), emb.sin()], dim=0).to(device) 
        return pe 

class RoPEMultiheadAttention(nn.Module):
    """
    Custom Multihead Attention (MHA) layer that applies Rotary Positional 
    Embedding (RoPE) to the Query (Q) and Key (K) vectors after projection.
    """
    def __init__(self, embed_dim, num_heads, dropout=0., batch_first=True):
        """
        Args:
            embed_dim (int): Total dimension of the input feature space.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projection for Q, K, V
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True) 
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = self.head_dim ** -0.5 # Attention scaling factor
        
        # Initialize RoPE generator
        self.rope_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len=4096) 
        
    def forward(self, query, key, value):
        """
        Applies RoPE to Q and K, calculates attention, and returns the output.
        """
        N, L_q, D = query.shape # Batch, Query Length, Dimension
        L_k = key.shape[1]      # Key Length
        is_cross_attn = (query is not key)
        
        # Project Query (Q), Key (K), and Value (V)
        qkv = self.in_proj(query)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Handle cross-attention case explicitly
        if is_cross_attn:
            kv_proj = self.in_proj(key)
            k_proj, v_proj, _ = kv_proj.chunk(3, dim=-1)
            k, v = k_proj, v_proj
            
        def reshape_heads(x):
            # Reshape from [N, L, E] to [N, H, L, D_H]
            return x.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)
        
        # Calculate RoPE embeddings
        seq_len = max(L_q, L_k)
        rot_emb_cos_sin = self.rope_emb(seq_len, query.device) 
        cos, sin = rot_emb_cos_sin.chunk(2, dim=0) 
        
        # Reshape for broadcasting: [1, 1, L, D_H]
        cos = cos.unsqueeze(1) 
        sin = sin.unsqueeze(1) 

        # Apply RoPE: q_rot = q * cos + rotate_half(q) * sin
        q = (q * cos[:, :, :L_q]) + (rotate_half(q) * sin[:, :, :L_q])
        k = (k * cos[:, :, :L_k]) + (rotate_half(k) * sin[:, :, :L_k])
        
        # Scaled Dot-Product Attention
        attn_output_weights = torch.matmul(q * self.scaling, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        attn_output = torch.matmul(attn_output_weights, v)
        
        # Reshape and project final output
        # [N, H, L, D_H] -> [N, L, H, D_H] -> [N, L, E]
        attn_output = attn_output.transpose(1, 2).flatten(start_dim=-2)
        output = self.out_proj(attn_output)
        
        # Return output and mean attention weights (for optional analysis)
        return output, attn_output_weights.mean(dim=1)

class RoPETransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer Encoder utilizing RoPEMultiheadAttention.
    Structure: Norm -> Self-Attention -> Residual -> Norm -> Feed-Forward -> Residual.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0., batch_first=True):
        """
        Args:
            d_model (int): The dimension of the input/output features.
            nhead (int): The number of attention heads.
            dim_feedforward (int): The hidden layer dimension of the FFN.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Position-wise Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.d_model = d_model

    def forward(self, src):
        """
        Args:
            src (torch.Tensor): Input sequence tensor.
            
        Returns:
            torch.Tensor: Output sequence tensor.
        """
        # Multi-Head Attention Block with Residual Connection
        src_norm = self.norm1(src)
        attn_out, _ = self.attn(src_norm, src_norm, src_norm)
        src = src + attn_out
        
        # Feed-Forward Network Block with Residual Connection
        src_norm = self.norm2(src)
        ffn_out = self.ffn(src_norm)
        src = src + ffn_out
        
        return src
    
    
class ResidualBlock(nn.Module):
    """
    Residual Block with Gated Fusion.
    Uses a trainable scalar weight (self.detail_weight) to scale the 
    high-frequency detail path output before merging with the main path.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        """
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            stride (int): Stride for the convolutional layers (for downsampling).
        """
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
        
        self.apply(init_weights) # Assuming init_weights is defined elsewhere

    def forward(self, x):
        """
        Computes the forward pass with Gated Fusion: 
        out = f(x) + (detail_weight * detail_path) + shortcut(x)
        """
        main_out = self.conv(x)
        detail_out = self.detail_conv(x)
        shortcut_out = self.shortcut(x)
        
        # Gated Fusion: Detail path output is scaled by the learned weight
        out = main_out + (self.detail_weight * detail_out) + shortcut_out
        
        out = self.gelu(out)
        return out

    
class ChannelReducer(nn.Module):
    """
    Reduces the number of channels using a simple 1x1 convolution, 
    followed by Batch Normalization and GELU activation.
    """
    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        self.apply(init_weights) # Assuming init_weights is defined elsewhere
        
    def forward(self, x):
        return self.conv(x)


class Unet_Enc(nn.Module):
    """
    U-Net Encoder with a dual-path input structure for FAF and Geometric features,
    and a Transformer layer with RoPE in the bottleneck region.
    """
    def __init__(self, args, img_channels=3, base_channels=16):
        """
        Args:
            args (object): Arguments including nhead, d_attn2.
            img_channels (int): Total input channels (3: FAF, Mask, Residual).
            base_channels (int): Base channel width (C).
        """
        super(Unet_Enc, self).__init__()
        C = base_channels 
        
        # 1. FAF Pathway (Input: 1 channel, Output: C/4 channels)
        self.conv1_faf = ResidualBlock(1, C // 4) 
        
        # 2. Geometric Pathway (Input: 2 channels, Output: 3C/4 channels)
        self.conv1_geo = ResidualBlock(2, C * 3 // 4) 
        
        # --- Base U-Net Path ---
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = ResidualBlock(C, C * 2) 
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = ResidualBlock(C * 2, C * 4)
        self.dropout3 = nn.Dropout2d(0.2) 
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = ResidualBlock(C * 4, C * 8)
        self.dropout4 = nn.Dropout2d(0.2) 
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # Transformer/Bottleneck setup
        self.te1_model_dim = C * 8
        self.te_sa4_1 = RoPETransformerEncoderLayer(
            d_model=self.te1_model_dim, 
            nhead=args.nhead, 
            dim_feedforward=args.d_attn2, 
            dropout=0, 
            batch_first=True
        )
        self.conv5 = ResidualBlock(C * 8, C * 16)

        # Skip Channel Reducers
        self.reduce_l1 = ChannelReducer(C, C // 2) 
        self.reduce_l2 = ChannelReducer(C * 2, C) 

        self.apply(init_weights)

    def forward(self, in_frames):
        N = in_frames.size(0)
        
        # Split input: (FAF, Mask+Residual)
        faf_input = in_frames[:, 0:1, :, :] 
        geo_input = in_frames[:, 1:3, :, :] 

        # --- L1 (256x256) - Dual Path Encoding ---
        feats_faf = self.conv1_faf(faf_input)       # Output C/4 channels
        feats_geo = self.conv1_geo(geo_input)       # Output 3C/4 channels
        
        # Merge the outputs along the Channel dimension
        feats1u_full = torch.cat((feats_faf, feats_geo), dim=1) # Total C channels
        
        # Calculate skip connection features and pool
        feats1u_reduced = self.reduce_l1(feats1u_full) # C -> C/2 channels
        x = self.pool1(feats1u_full) # Use FULL feature for max-pooling
        
        # --- L2 (128x128) and onward ---
        feats2u_full = self.conv2(x)
        feats2u_reduced = self.reduce_l2(feats2u_full)
        x = self.pool2(feats2u_full)
        
        feats3u = self.conv3(x)
        feats3u = self.dropout3(feats3u)
        x_pre_attn = self.pool3(feats3u) 
        feats4u_pre = self.conv4(x_pre_attn)
        feats4u_pre = self.dropout4(feats4u_pre)

        # Transformer/Bottleneck
        tokens4 = feats4u_pre.flatten(2).transpose(1, 2) # [N, H*W, D]
        q1 = self.te_sa4_1(tokens4) 
        # Reshape back to 4D and apply residual connection
        q1_4d = q1.transpose(1, 2).reshape(N, self.te1_model_dim, *x_pre_attn.shape[2:])
        feats4u = feats4u_pre + q1_4d
        
        x_bottleneck_pre = self.pool4(feats4u)
        bottleneck_4d = self.conv5(x_bottleneck_pre) 
        
        # Return skip features and bottleneck output
        return feats1u_reduced, feats2u_reduced, feats3u, feats4u, bottleneck_4d
 
# class Unet_Enc(nn.Module):
#     """
#     U-Net Encoder with a dual-path input structure, and Global Spatial Attention
#     at both L4 (32x32) and L5 (16x16) to enhance feature fidelity.
#     """
#     def __init__(self, args, img_channels=3, base_channels=16):
#         super(Unet_Enc, self).__init__()
#         C = base_channels
        
#         # 1. FAF Pathway
#         self.conv1_faf = ResidualBlock(1, C // 4)
        
#         # 2. Geometric Pathway
#         self.conv1_geo = ResidualBlock(2, C * 3 // 4)
        
#         # --- Base U-Net Path ---
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
        
#         self.conv2 = ResidualBlock(C, C * 2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
        
#         self.conv3 = ResidualBlock(C * 2, C * 4)
#         self.dropout3 = nn.Dropout2d(0.2)
#         self.pool3 = nn.MaxPool2d(kernel_size=2)
        
#         # L4 Attention Setup
#         self.te1_model_dim = C * 8 # 128
#         self.conv4 = ResidualBlock(C * 4, C * 8)
#         self.dropout4 = nn.Dropout2d(0.2)
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
        
#         # L4 Spatial Attention (32x32)
#         self.te_sa4_1 = RoPETransformerEncoderLayer(
#             d_model=self.te1_model_dim,
#             nhead=args.nhead,
#             dim_feedforward=args.d_attn2,
#             dropout=0,
#             batch_first=True
#         )
        
#         # L5 Bottleneck Setup
#         self.te2_model_dim = C * 16 # 256
#         self.conv5 = ResidualBlock(C * 8, C * 16)
        
#         # L5 Spatial Attention (16x16) - NEW ADDITION
#         self.te_sa5_1 = RoPETransformerEncoderLayer(
#             d_model=self.te2_model_dim,
#             nhead=args.nhead,
#             dim_feedforward=args.d_attn2,
#             dropout=0,
#             batch_first=True
#         )

#         # Skip Channel Reducers
#         self.reduce_l1 = ChannelReducer(C, C // 2)
#         self.reduce_l2 = ChannelReducer(C * 2, C)

#         self.apply(init_weights)

#     def forward(self, in_frames):
#         N = in_frames.size(0)
        
#         # Split input
#         faf_input = in_frames[:, 0:1, :, :]
#         geo_input = in_frames[:, 1:3, :, :]

#         # --- L1 (256x256) ---
#         feats_faf = self.conv1_faf(faf_input)
#         feats_geo = self.conv1_geo(geo_input)
#         feats1u_full = torch.cat((feats_faf, feats_geo), dim=1)
#         feats1u_reduced = self.reduce_l1(feats1u_full)
#         x = self.pool1(feats1u_full)

#         # --- L2 (128x128) ---
#         feats2u_full = self.conv2(x)
#         feats2u_reduced = self.reduce_l2(feats2u_full)
#         x = self.pool2(feats2u_full)
        
#         # --- L3 (64x64) ---
#         feats3u = self.conv3(x)
#         feats3u = self.dropout3(feats3u)
#         x_pre_attn4 = self.pool3(feats3u)
        
#         # --- L4 (32x32) with Spatial Attention ---
#         feats4u_pre = self.conv4(x_pre_attn4)
#         feats4u_pre = self.dropout4(feats4u_pre)
        
#         # L4 Attention
#         tokens4 = feats4u_pre.flatten(2).transpose(1, 2) # [N, 32*32, 128]
#         q4 = self.te_sa4_1(tokens4)
#         q4_4d = q4.transpose(1, 2).reshape(N, self.te1_model_dim, *x_pre_attn4.shape[2:])
#         feats4u = feats4u_pre + q4_4d # Residual Connection
        
#         x_bottleneck_pre = self.pool4(feats4u)

#         # --- L5 (16x16) Bottleneck with Spatial Attention ---
#         bottleneck_conv = self.conv5(x_bottleneck_pre) # [N, 256, 16, 16]

#         # L5 Attention (NEW)
#         tokens5 = bottleneck_conv.flatten(2).transpose(1, 2) # [N, 16*16, 256]
#         q5 = self.te_sa5_1(tokens5)
#         q5_4d = q5.transpose(1, 2).reshape(N, self.te2_model_dim, *x_bottleneck_pre.shape[2:])
#         bottleneck_4d = bottleneck_conv + q5_4d # Residual Connection

#         return feats1u_reduced, feats2u_reduced, feats3u, feats4u, bottleneck_4d

class Unet_Dec(nn.Module):
    """
    U-Net Decoder path with additive skip connections (Levels 1 & 2) 
    and a final convolutional layer with Sigmoid activation.
    """
    def __init__(self, args, img_channels=3, base_channels=16):
        """
        Args:
            args (object): Arguments including nhead, d_attn1, d_attn2.
            img_channels (int): Number of final output channels (3).
            base_channels (int): Base channel width (C).
        """
        super(Unet_Dec, self).__init__()
        C = base_channels 

        # --- L4 Setup (C=16 -> 8C=128) ---
        self.upconv4 = nn.ConvTranspose2d(C * 16, C * 8, kernel_size=2, stride=2)
        # Input channel count: Upconv output (C*8) + Skip connection (C*8) = C*16
        self.up_conv4 = ResidualBlock(C * 8 + C * 8, C * 8)
        
        # Optional Transformer layers (defined but not fully utilized in this snippet's forward)
        self.te2d_model_dim = C * 8
        self.te_sa4d_1 = RoPETransformerEncoderLayer(d_model=self.te2d_model_dim, nhead=args.nhead, dim_feedforward=args.d_attn2, dropout=0, batch_first=True)
        self.mha4_ca = RoPEMultiheadAttention(embed_dim=self.te2d_model_dim, num_heads=args.nhead, dropout=0, batch_first=True) 

        # --- L3 Setup (C=128 -> 4C=64) ---
        self.upconv3 = nn.ConvTranspose2d(C * 8, C * 4, kernel_size=2, stride=2)
        # Input channel count: Upconv output (C*4) + Skip connection (C*4) = C*8
        self.up_conv3 = ResidualBlock(C * 4 + C * 4, C * 4)

        self.te1d_model_dim = C * 4
        self.te_sa3d_1 = RoPETransformerEncoderLayer(d_model=self.te1d_model_dim, nhead=args.nhead, dim_feedforward=args.d_attn1, dropout=0, batch_first=True)
        self.mha3_ca = RoPEMultiheadAttention(embed_dim=self.te1d_model_dim, num_heads=args.nhead, dropout=0, batch_first=True) 

        # --- L2 Setup (ADDITIVE Skip Connection) ---
        # Upconv output: C*2. Reduced Skip input: C. Input to up_conv2 is C*2.
        self.upconv2 = nn.ConvTranspose2d(C * 4, C * 2, kernel_size=2, stride=2) 
        # Match reduced skip to upconv output channels
        self.l2_add_match = nn.Conv2d(C, C * 2, kernel_size=1) 
        self.up_conv2 = ResidualBlock(C * 2, C * 2) 

        # --- L1 Setup (ADDITIVE Skip Connection) ---
        # Upconv output: C. Reduced Skip input: C/2. Input to up_conv1 is C.
        self.upconv1 = nn.ConvTranspose2d(C * 2, C, kernel_size=2, stride=2) 
        # Match reduced skip to upconv output channels
        self.l1_add_match = nn.Conv2d(C // 2, C, kernel_size=1) 
        self.up_conv1 = ResidualBlock(C, C) 
        
        # Final output layer
        self.final_conv = nn.Conv2d(C, img_channels, kernel_size=1)
        self.sig = nn.Sigmoid()

        self.apply(init_weights) 
        
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
        # Match skip channels to upconv output channels
        skip_matched = self.l2_add_match(feats2u_enc)
        x = x + skip_matched
        feats2d = self.up_conv2(x)

        # Level 1: Upsample and ADDITIVE Skip
        x = self.upconv1(feats2d)
        # Match skip channels to upconv output channels
        skip_matched = self.l1_add_match(feats1u_enc)
        x = x + skip_matched
        feats1d = self.up_conv1(x)
        
        # Final output
        out_frames_logits = self.final_conv(feats1d)
        out_frames = self.sig(out_frames_logits)

        return out_frames
    
    
class U_Net_AE(nn.Module):
    """ 
    Wrapper combining the U-Net Encoder (Unet_Enc) and the Decoder (Unet_Dec) 
    to form the complete Auto-encoder architecture. 
    """
    def __init__(self, args, img_channels=3, base_channels=16): # Assumes BASE_CHANNELS is 16
        """
        Args:
            args (object): Configuration arguments.
            img_channels (int): Input/Output channel count (3).
            base_channels (int): Base channel width (C).
        """
        super(U_Net_AE, self).__init__()
        self.E1 = Unet_Enc(args, img_channels, base_channels)
        self.D1 = Unet_Dec(args, img_channels, base_channels)
        
    def forward(self, in_frames):
        """
        Performs the full auto-encoding process.
        
        Returns:
            tuple: (out_frames, skip_features...)
        """
        feats1u, feats2u, feats3u, feats4u, bottleneck_4d = self.E1(in_frames)
        out_frames = self.D1(feats1u, feats2u, feats3u, feats4u, bottleneck_4d)
        
        # Return reconstruction and all skip features for potential multi-task loss calculation
        return out_frames, feats1u, feats2u, feats3u, feats4u, bottleneck_4d
