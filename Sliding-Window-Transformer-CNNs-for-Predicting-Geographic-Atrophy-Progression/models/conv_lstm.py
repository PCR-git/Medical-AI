
import torch
import torch.nn as nn
import torch.nn.functional as F 

from .autoencoder import Unet_Enc, Unet_Dec 
from .model_utils import init_weights
from .CFB import FusionBlockBottleneck, ChannelFusionBlock
from .cnn_ablations import CNN_Unet_Enc, CNN_Unet_Dec

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell enhanced with a direct skip connection (residual connection) 
    from the input (X_t) to the output (H_t) to stabilize spatial features.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

        # 1x1 convolution to match input (X_t) channels to hidden (H_t) channels for the residual skip
        if input_dim != hidden_dim:
            self.input_match_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=False)
        else:
            self.input_match_conv = nn.Identity()

    def forward(self, input_tensor, cur_state):
        h_prev, c_prev = cur_state
        
        combined = torch.cat([input_tensor, h_prev], dim=1)  
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(combined_conv, 4, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_prev + i * g
        h_base = o * torch.tanh(c_next)
        
        # --- Residual Skip Connection ---
        # Match the input tensor dimensions to the hidden state dimensions (H_t) and add them.
        h_skip = self.input_match_conv(input_tensor)
        h_next = h_base + h_skip # Add the current input to the evolved state

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        H, W = image_size
        device = next(self.parameters()).device
        
        h_state = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        c_state = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        
        return (h_state, c_state)
 
# -----------------------------------------------------------------------------------------

## Larger version
# class ConvLSTMCore(nn.Module):
#     """
#     Multi-scale recurrent core using ConvLSTM with a 50% channel bottleneck 
#     applied to deep layers (L3, L4, BN) for parameter reduction and stability.
#     """
#     def __init__(self, base_channels, lstm_kernel_size=3):
#         super(ConvLSTMCore, self).__init__()
#         C = base_channels # 16
#         REDUCTION = 2 
        
#         # INCREASED CAPACITY FACTOR (3/4)
#         FACTOR_NUMERATOR = 3 
#         FACTOR_DENOMINATOR = 4
        
#         # L1 and L2 (Shallow: Full Channels)
#         self.lstm_l1 = ConvLSTMCell(C // 2, C // 2, lstm_kernel_size, bias=True)
#         self.lstm_l2 = ConvLSTMCell(C, C, lstm_kernel_size, bias=True)

#         # L3: Input 4C (64) -> Hidden 3C (48)
#         C_L3_H = C * 4 * FACTOR_NUMERATOR // FACTOR_DENOMINATOR
#         self.proj_l3_in = nn.Conv2d(C * 4, C_L3_H, kernel_size=1) 
#         self.lstm_l3 = ConvLSTMCell(C_L3_H, C_L3_H, lstm_kernel_size, bias=True)
#         self.proj_l3_out = nn.Conv2d(C_L3_H, C * 4, kernel_size=1) 

#         # L4: Input 8C (128) -> Hidden 6C (96)
#         C_L4_H = C * 8 * FACTOR_NUMERATOR // FACTOR_DENOMINATOR
#         self.proj_l4_in = nn.Conv2d(C * 8, C_L4_H, kernel_size=1) 
#         self.lstm_l4 = ConvLSTMCell(C_L4_H, C_L4_H, lstm_kernel_size, bias=True)
#         self.proj_l4_out = nn.Conv2d(C_L4_H, C * 8, kernel_size=1)

#         # BN: Input 16C (256) -> Hidden 12C (192)
#         C_BN_H = C * 16 * FACTOR_NUMERATOR // FACTOR_DENOMINATOR
#         self.proj_bn_in = nn.Conv2d(C * 16, C_BN_H, kernel_size=1) 
#         self.lstm_bn = ConvLSTMCell(C_BN_H, C_BN_H, lstm_kernel_size, bias=True)
#         self.proj_bn_out = nn.Conv2d(C_BN_H, C * 16, kernel_size=1) 

#         self.lstms = [self.lstm_l1, self.lstm_l2, self.lstm_l3, self.lstm_l4, self.lstm_bn]
        
#         # Define projection pairs: (proj_in, proj_out)
#         self.projs = [(None, None), (None, None), 
#                       (self.proj_l3_in, self.proj_l3_out), 
#                       (self.proj_l4_in, self.proj_l4_out), 
#                       (self.proj_bn_in, self.proj_bn_out)]
        
#         self.apply(init_weights) 

#     def forward(self, encoder_features_flat, B, T_pred):
        
#         # 1. Prepare Features and States
#         features_by_time = []
#         for t in range(T_pred):
#             F_t = [f[t*B:(t+1)*B] for f in encoder_features_flat]
#             features_by_time.append(F_t)
        
#         feature_sizes = [(f.shape[2], f.shape[3]) for f in features_by_time[0]]
#         hidden_states = [lstm.init_hidden(B, feature_sizes[i]) 
#                          for i, lstm in enumerate(self.lstms)]

#         # 2. Sequence Iteration and Prediction 
#         evolved_features_flat = [[] for _ in range(5)]
        
#         for t in range(T_pred):
#             F_t_levels = features_by_time[t] 

#             for l in range(5):
#                 lstm = self.lstms[l]
#                 proj_in, proj_out = self.projs[l]
#                 F_t = F_t_levels[l]
#                 h_state, c_state = hidden_states[l]
                
#                 # --- Bottleneck Input ---
#                 if proj_in is not None:
#                     F_t = proj_in(F_t) 
                    
#                 # Core LSTM Computation (Uses the skip connection implemented inside the cell)
#                 h_next, c_next = lstm(F_t, (h_state, c_state))
                
#                 # --- Bottleneck Output ---
#                 h_output = proj_out(h_next) if proj_out is not None else h_next 
                
#                 hidden_states[l] = (h_next, c_next) # Store the state for recurrence
#                 evolved_features_flat[l].append(h_output)
                
#         # 3. Final Output Compilation
#         evolved_output = [torch.cat(E_level, dim=0) for E_level in evolved_features_flat]
        
#         return tuple(evolved_output)


# Smaller version
class ConvLSTMCore(nn.Module):
    """
    Multi-scale recurrent core using ConvLSTM, modified to use a 5/8 channel bottleneck 
    on deep layers (L3, L4, BN) to precisely tune the parameter count to ~6.5M.
    """
    def __init__(self, base_channels, lstm_kernel_size=3):
        super(ConvLSTMCore, self).__init__()
        C = base_channels # 16
        
        # --- MODIFIED FACTOR: 5/8 Channel Reduction ---
        FACTOR_NUMERATOR = 5
        FACTOR_DENOMINATOR = 8
        
        # L1 and L2 (Shallow: Full Channels) - NO CHANGE (4.5M core was sufficient here)
        self.lstm_l1 = ConvLSTMCell(C // 2, C // 2, lstm_kernel_size, bias=True)
        self.lstm_l2 = ConvLSTMCell(C, C, lstm_kernel_size, bias=True)

        # L3: Input 4C (64) -> Hidden 2.5C (40)
        C_L3_H = C * 4 * FACTOR_NUMERATOR // FACTOR_DENOMINATOR # 64 * 5/8 = 40
        self.proj_l3_in = nn.Conv2d(C * 4, C_L3_H, kernel_size=1) 
        self.lstm_l3 = ConvLSTMCell(C_L3_H, C_L3_H, lstm_kernel_size, bias=True)
        self.proj_l3_out = nn.Conv2d(C_L3_H, C * 4, kernel_size=1) 

        # L4: Input 8C (128) -> Hidden 5C (80)
        C_L4_H = C * 8 * FACTOR_NUMERATOR // FACTOR_DENOMINATOR # 128 * 5/8 = 80
        self.proj_l4_in = nn.Conv2d(C * 8, C_L4_H, kernel_size=1) 
        self.lstm_l4 = ConvLSTMCell(C_L4_H, C_L4_H, lstm_kernel_size, bias=True)
        self.proj_l4_out = nn.Conv2d(C_L4_H, C * 8, kernel_size=1)

        # BN: Input 16C (256) -> Hidden 10C (160)
        C_BN_H = C * 16 * FACTOR_NUMERATOR // FACTOR_DENOMINATOR # 256 * 5/8 = 160
        self.proj_bn_in = nn.Conv2d(C * 16, C_BN_H, kernel_size=1) 
        self.lstm_bn = ConvLSTMCell(C_BN_H, C_BN_H, lstm_kernel_size, bias=True)
        self.proj_bn_out = nn.Conv2d(C_BN_H, C * 16, kernel_size=1) 

        self.lstms = [self.lstm_l1, self.lstm_l2, self.lstm_l3, self.lstm_l4, self.lstm_bn]
        
        # Define projection pairs: (proj_in, proj_out)
        self.projs = [(None, None), (None, None), 
                      (self.proj_l3_in, self.proj_l3_out), 
                      (self.proj_l4_in, self.proj_l4_out), 
                      (self.proj_bn_in, self.proj_bn_out)]
        
        self.apply(init_weights) 

    def forward(self, encoder_features_flat, B, T_pred):
        
        # 1. Prepare Features and States
        features_by_time = []
        for t in range(T_pred):
            F_t = [f[t*B:(t+1)*B] for f in encoder_features_flat]
            features_by_time.append(F_t)
        
        feature_sizes = [(f.shape[2], f.shape[3]) for f in features_by_time[0]]
        hidden_states = [lstm.init_hidden(B, feature_sizes[i]) 
                         for i, lstm in enumerate(self.lstms)]

        # 2. Sequence Iteration and Prediction 
        evolved_features_flat = [[] for _ in range(5)]
        
        for t in range(T_pred):
            F_t_levels = features_by_time[t] 

            for l in range(5):
                lstm = self.lstms[l]
                proj_in, proj_out = self.projs[l]
                F_t = F_t_levels[l]
                h_state, c_state = hidden_states[l]
                
                # --- Bottleneck Input ---
                if proj_in is not None:
                    F_t = proj_in(F_t) 
                    
                # Core LSTM Computation (Uses the skip connection implemented inside the cell)
                h_next, c_next = lstm(F_t, (h_state, c_state))
                
                # --- Bottleneck Output ---
                h_output = proj_out(h_next) if proj_out is not None else h_next 
                
                hidden_states[l] = (h_next, c_next) # Store the state for recurrence
                evolved_features_flat[l].append(h_output)
                
        # 3. Final Output Compilation
        evolved_output = [torch.cat(E_level, dim=0) for E_level in evolved_features_flat]
        
        return tuple(evolved_output)
    
# -----------------------------------------------------------------------------------------
    
    
# class ConvLSTMBaseline(nn.Module):
#     """
#     A baseline model using a multi-scale ConvLSTM core for temporal prediction.
#     It replaces the complex CFB/SWA/DynNet system while reusing the standard U-Net structure.
#     """
#     # FIX: Set external dependencies to None as defaults
#     def __init__(self, args, img_channels=None, base_channels=None):
#         super(ConvLSTMBaseline, self).__init__()
        
#         # FIX: Resolve dependencies inside the constructor
#         if img_channels is None:
#             # Assumes args.img_channels is available during call time
#             img_channels = args.img_channels
#         if base_channels is None:
#             # Assumes BASE_CHANNELS constant is imported or default is used (e.g., 16)
#             base_channels = BASE_CHANNELS
        
#         # E1: Feature Extractor (Encoder) - Uses the existing Unet_Enc architecture
#         self.E1 = Unet_Enc(args, img_channels, base_channels)
        
#         # P: Temporal Core - Replaces CFB/SWA/DynNet with a single recurrent module
#         self.P_LSTM = ConvLSTMCore(base_channels, lstm_kernel_size=5)
        
#         # D1: Frame Reconstructor (Decoder) - Uses the existing Unet_Dec architecture
#         self.D1 = Unet_Dec(args, img_channels, base_channels)
        
#     def forward(self, input_clips):
#         """
#         Processes a clip (I0, I1, I2, I3) and returns predictions (I1, I2, I3) 
#         and feature maps for loss calculation.
#         """
#         B, C, T_total, H, W = input_clips.shape
#         T_pred = T_total - 1 # 3 predicted frames (I1, I2, I3)
        
#         # --- A. ENCODER INPUT PREPARATION ---
        
#         # Input frames for E1 (I0, I1, I2): (N*T_pred, C, H, W)
#         input_frames_E1 = input_clips[:, :, :T_pred, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
#         # Ground Truth for Anchor (I0)
#         I0_gt = input_clips[:, :, 0, :, :] 
        
#         # --- B. FEATURE EXTRACTION (E1) ---
#         # E1_features_flat is a tuple of 5 feature tensors (F1, F2, F3, F4, BN), shape [B * T_pred, ...]
#         E1_features_flat = self.E1(input_frames_E1)
        
#         # --- C. RECURRENT TEMPORAL PREDICTION (P_LSTM) ---
#         # The LSTM processes the sequence of features (F0, F1, F2) to output the 
#         # evolved features (E1, E2, E3). 
#         # Output is a tuple of 5 evolved features, shape [B*T_pred, ...]
#         Evolved_flat = self.P_LSTM(E1_features_flat, B, T_pred)
        
#         # Unpack evolved features: E_bn is the bottleneck state
#         E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = Evolved_flat
        
#         # --- D. ANCHOR RECONSTRUCTION (I_hat_0) ---
#         # Features for I0 are the first B items in E1_features_flat
#         I0_E1_features = [f[:B] for f in E1_features_flat]
#         I0_hat = self.D1(*I0_E1_features)

#         # --- E. DECODING AND RESHAPING ---
        
#         # Decode Evolved Features (E1, E2, E3)
#         out_frames_pred = self.D1(*Evolved_flat)
        
#         # Reshape prediction output back to clip format: (N, C, T, H, W)
#         predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        
#         # Targets are correctly sliced to (N, C, T, H, W)
#         targets = input_clips[:, :, 1:, :, :]
        
#         # Returns 9 items for loss tracking, identical to SWAU_Net output
#         return predictions, targets, I0_hat, I0_gt, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved


class ConvLSTMBaseline(nn.Module):
    """
    A strong recurrent baseline that uses the U-Net structure and a multi-scale
    ConvLSTM core, augmented with Channel Fusion Blocks (CFB) for pre- and post-
    recurrent feature refinement, matching the complexity structure of SWAU_Net.
    
    This baseline tests standard recurrence against SWA's explicit context aggregation.
    """
    # Use None as default in the signature to prevent BASE_CHANNELS NameError on import
    def __init__(self, args, img_channels=None, base_channels=None):
        super().__init__()
        
        # Safely resolve channel values inside the function body
        base_channels = base_channels if base_channels is not None else BASE_CHANNELS
        img_channels = img_channels if img_channels is not None else 3 # Default to 3 for standard color/FAF
        
        # E1: Feature Extractor (Encoder)
        self.E1 = Unet_Enc(args, img_channels, base_channels)
        
        # CFB_enc: Pre-Recurrence Feature Refinement (New Addition)
        self.CFB_enc = ChannelFusionBlock(base_channels) 
        
        # P: Temporal Core - Recurrent prediction module
        self.P_LSTM = ConvLSTMCore(base_channels, lstm_kernel_size=5)
        
        # CFB_dec: Post-Recurrence Feature Refinement (New Addition)
        self.CFB_dec = ChannelFusionBlock(base_channels)
        
        # D1: Frame Reconstructor (Decoder)
        self.D1 = Unet_Dec(args, img_channels, base_channels)
        
        # NOTE: init_weights assumed to be called via external script or wrapper
        # self.apply(init_weights) 

    def forward(self, input_clips):
        """
        Processes a clip (I0, I1, I2, I3) and returns predictions (I1, I2, I3) 
        with CFB integration.
        """
        B, C, T_total, H, W = input_clips.shape
        T_pred = T_total - 1 # e.g., 3 predicted frames (I1, I2, I3)
        
        # --- A. ENCODER INPUT PREPARATION ---
        
        # Input frames for E1 (I0, I1, I2): (N*T_pred, C, H, W)
        input_frames_E1 = input_clips[:, :, :T_pred, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
        # Ground Truth for Anchor (I0)
        I0_gt = input_clips[:, :, 0, :, :] 
        
        # --- B. FEATURE EXTRACTION (E1) ---
        # E1_features_flat: raw features F for I0, I1, I2. Shape [B * T_pred, ...]
        E1_features_flat = self.E1(input_frames_E1)

        # --- C. PRE-RECURRENCE CHANNEL FUSION (CFB_enc) ---
        # F' = CFB_enc(F). Refined features before LSTM.
        F_refined_flat = self.CFB_enc(E1_features_flat)
        
        # --- D. ANCHOR RECONSTRUCTION (I_hat_0) ---
        # Features for I0 are the first B items in the refined features F'
        I0_F_refined = [f[:B] for f in F_refined_flat]
        I0_hat = self.D1(*I0_F_refined)

        # --- E. RECURRENT TEMPORAL PREDICTION (P_LSTM) ---
        # H = P_LSTM(F'). The LSTM processes the sequence of refined features (F'0, F'1, F'2) 
        # to output the hidden states (H1, H2, H3).
        H_evolved_flat = self.P_LSTM(F_refined_flat, B, T_pred)

        # --- F. POST-RECURRENCE CHANNEL FUSION (CFB_dec) ---
        # E = CFB_dec(H). Polished features E for decoding.
        Evolved_polished = self.CFB_dec(H_evolved_flat)

        E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = Evolved_polished

        # --- G. DECODING AND RESHAPING ---
        
        # Decode Evolved Features (E1, E2, E3)
        out_frames_pred = self.D1(*Evolved_polished)
        
        # Reshape prediction output back to clip format: (N, C, T, H, W)
        predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        
        # Targets are correctly sliced to (N, C, T_pred, H, W)
        targets = input_clips[:, :, 1:, :, :]
        
        # Return 9 items for consistent loss tracking across all models
        return predictions, targets, I0_hat, I0_gt, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved
    
    
# ---------------------------------------------------------------------------------


class ConvLSTM_Simple(nn.Module):
    """
    A foundational recurrent baseline using simplified CNN-based U-Net 
    components and removing all Channel Fusion Blocks (CFB) to test the 
    raw performance of ConvLSTM recurrence on minimally processed features.
    """
    # Use None as default in the signature
    def __init__(self, args, img_channels=None, base_channels=None):
        super().__init__()
        
        # Safely resolve channel values inside the function body
        # NOTE: BASE_CHANNELS must be defined/accessible globally
        base_channels = base_channels if base_channels is not None else BASE_CHANNELS 
        img_channels = img_channels if img_channels is not None else 3 
        
        # E1: Feature Extractor (Using simplified CNN U-Net Encoder)
        self.E1 = CNN_Unet_Enc(args, img_channels, base_channels) 
        
        # CFB_enc: REMOVED
        # self.CFB_enc = ChannelFusionBlock(base_channels) 
        
        # P: Temporal Core - Recurrent prediction module
        self.P_LSTM = ConvLSTMCore(base_channels, lstm_kernel_size=5)
        
        # CFB_dec: REMOVED
        # self.CFB_dec = ChannelFusionBlock(base_channels)
        
        # D1: Frame Reconstructor (Using simplified CNN U-Net Decoder)
        self.D1 = CNN_Unet_Dec(args, img_channels, base_channels)
        
        # NOTE: init_weights assumed to be called via external script or wrapper

    def forward(self, input_clips):
        """
        Processes a clip (I0, I1, I2, I3) using the flow: 
        E1 -> P_LSTM -> D1 (No CFB layers).
        """
        B, C, T_total, H, W = input_clips.shape
        T_pred = T_total - 1 # e.g., 3 predicted frames (I1, I2, I3)
        
        # --- A. ENCODER INPUT PREPARATION ---
        input_frames_E1 = input_clips[:, :, :T_pred, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        I0_gt = input_clips[:, :, 0, :, :] 
        
        # --- B. FEATURE EXTRACTION (E1) ---
        # E1_features_flat: raw features F for I0, I1, I2. Shape [B * T_pred, ...]
        E1_features_flat = self.E1(input_frames_E1)

        # --- C. PRE-RECURRENCE PATH (CFB_enc REMOVED) ---
        # F' = F. Features are used directly from E1 without refinement.
        F_input_flat = E1_features_flat 
        
        # --- D. ANCHOR RECONSTRUCTION (I_hat_0) ---
        # Features for I0 are the first B items in the raw features F
        I0_F_raw = [f[:B] for f in F_input_flat]
        I0_hat = self.D1(*I0_F_raw)

        # --- E. RECURRENT TEMPORAL PREDICTION (P_LSTM) ---
        # H = P_LSTM(F). The LSTM processes the sequence of raw features.
        # H_evolved_flat is the output hidden state sequence.
        H_evolved_flat = self.P_LSTM(F_input_flat, B, T_pred)

        # --- F. POST-RECURRENCE PATH (CFB_dec REMOVED) ---
        # E = H. Evolved features are used directly without polishing.
        Evolved_features = H_evolved_flat

        E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = Evolved_features

        # --- G. DECODING AND RESHAPING ---
        out_frames_pred = self.D1(*Evolved_features)
        
        # Reshape prediction output back to clip format: (N, C, T, H, W)
        predictions = out_frames_pred.reshape(B, T_pred, C, H, W).permute(0, 2, 1, 3, 4)
        
        # Targets are correctly sliced to (N, C, T_pred, H, W)
        targets = input_clips[:, :, 1:, :, :]
        
        # Return 9 items for consistent loss tracking across all models
        return predictions, targets, I0_hat, I0_gt, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved
    