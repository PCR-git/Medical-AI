
import torch
import numpy as np
from tqdm import tqdm

from augmentation_utils import f_augment_spatial_and_intensity 


# --- BATCH NORM FREEZING ---
def freeze_batch_norm(model):
    """Sets all BatchNorm layers to evaluation mode to freeze running statistics."""
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


def f_single_epoch_AE(full_clean_data_tensor_cpu, model, optimizer, loss_fn_bce, loss_fn_dice, loss_fn_gdl, loss_fn_l1, loss_fn_l2, args, batch_size,
                      lambda_gdl=1e-3, lambda_residual=5.0, lambda_llr=1e-5, lambda_bottleneck=1e-6):
    """ 
    Executes a single training epoch for the Autoencoder model using a custom 
    multi-channel loss mix via manual batch iteration.
    
    Args:
        full_clean_data_tensor_cpu (torch.Tensor): The entire training dataset tensor (on CPU).
        model (nn.Module): The Autoencoder model (U_Net_AE).
        optimizer (Optimizer): PyTorch optimizer.
        loss_fn_bce (nn.Module): Binary Cross-Entropy loss function.
        loss_fn_dice (function): Dice loss function.
        loss_fn_gdl (nn.Module): Gradient Difference Loss function.
        loss_fn_l1 (nn.Module): L1 (MAE) loss function.
        loss_fn_l2 (nn.Module): L2 (MSE) loss function.
        args (object): Configuration arguments (must include args.device).
        batch_size (int): The number of samples to process per iteration.
        lambda_* (float): Weight multipliers for the corresponding loss terms.
        
    Returns:
        np.array: Array of total loss values recorded per batch iteration.
    """
    model.train()
    epoch_losses = []

    # --- Setup Manual Iteration ---
    N_SAMPLES = len(full_clean_data_tensor_cpu)
    N_BATCHES = N_SAMPLES // batch_size
    
    # Shuffle indices once per epoch to mimic DataLoader shuffle
    indices = torch.randperm(N_SAMPLES).tolist()
    
    # Define Channel Weights for Primary Losses
    lambda_faf = 0.2
    lambda_mask = 1.0
    
    # --- Manual Batch Loop ---
    for i in tqdm(range(N_BATCHES), desc="Batch progress"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        # Get the global (shuffled) indices for this batch
        batch_indices = indices[start_idx:end_idx]
        
        # Fetch data slice from CPU tensor and move to GPU
        # data_batch shape: (B, C=3, T=4, H, W)
        data_batch = full_clean_data_tensor_cpu[batch_indices].to(args.device)
        
        B, C, T, H, W = data_batch.shape
        
        # Flatten the data to treat all T=4 time steps as independent samples for the AE
        # in_frames shape: [B * T, C, H, W]
        in_frames = data_batch.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Target is the input itself (Autoencoder operation)
        target_frames = in_frames 

        # --- Single Iteration Logic ---
        optimizer.zero_grad()
        # The model returns 6 outputs: out_frames and 5 skip/bottleneck feature maps
        out_frames, feats1u, feats2u, feats3u, feats4u, bottleneck_4d = model(in_frames)

        # --- Multi-Channel Loss Calculation ---
        
        # Split Tensors by channel: [B*T, C, H, W] -> [B*T, 1, H, W]
        target_faf = target_frames[:, 0:1, :, :]
        out_faf = out_frames[:, 0:1, :, :]
        target_mask = target_frames[:, 1:2, :, :]
        out_mask = out_frames[:, 1:2, :, :]
        target_residual = target_frames[:, 2:3, :, :]
        out_residual = out_frames[:, 2:3, :, :]
        
        # 1. FAF Loss (Weighted BCE + GDL)
        loss_faf_bce = loss_fn_bce(out_faf, target_faf)
        loss_faf_gdl = loss_fn_gdl(out_faf, target_faf)
        
        # 2. Mask Loss (Dice + GDL)
        loss_mask_dice = loss_fn_dice(target_mask, out_mask)
        loss_mask_gdl = loss_fn_gdl(out_mask, target_mask)

        # 3. Residual Loss (Weighted Dice + GDL)
        loss_residual_dice = loss_fn_dice(target_residual, out_residual)
        loss_residual_gdl = loss_fn_gdl(out_residual, target_residual)
        
        # 4. Low-Level Regularization (LLR) Loss (L1 on high-res feature maps)
        loss_llr = (
            loss_fn_l1(feats1u, torch.zeros_like(feats1u)) +
            loss_fn_l1(feats2u, torch.zeros_like(feats2u)) +
            loss_fn_l1(feats3u, torch.zeros_like(feats3u))
        )
        
        # 5. Bottleneck L2 Regularization (Penalizes large feature values)
        loss_bottleneck_l2 = loss_fn_l2(bottleneck_4d, torch.zeros_like(bottleneck_4d))
        
        # Total Loss: Weighted combination of all terms
        total_loss = (
            (lambda_faf * loss_faf_bce) + 
            (lambda_mask * loss_mask_dice) + 
            (lambda_residual * loss_residual_dice) + 
            (lambda_gdl * (loss_faf_gdl + loss_mask_gdl + loss_residual_gdl)) +
            (lambda_llr * loss_llr) + 
            (lambda_bottleneck * loss_bottleneck_l2)
        )

        # Backward Pass and Optimization
        total_loss.backward()
        optimizer.step()
        
        epoch_losses.append(total_loss.item())
    
    # Note dropped samples from the end of the dataset
    N_DROPPED = N_SAMPLES % batch_size
    if N_DROPPED > 0:
        print(f"Note: Dropped {N_DROPPED} samples in the final partial batch.")
        
    return np.array(epoch_losses)

# ----------------------------------------------------------------------------------------------------------------------

def f_single_epoch_spatiotemporal(full_clean_data_tensor_cpu, model, optimizer, loss_fn_bce, loss_fn_dice, loss_fn_gdl, loss_fn_l1, loss_fn_l2, args, batch_size,
                            lambda_gdl=1e-3, lambda_faf=0.2, lambda_mask=1.0, lambda_residual=5.0, lambda_recon=0.1, lambda_llr=1e-5, lambda_bottleneck=1e-6, use_augmentation=False):
    """ 
    Executes a single training epoch for the UPredNet model, including prediction, 
    reconstruction, and regularization losses.
    
    Args:
        full_clean_data_tensor_cpu (torch.Tensor): The entire training dataset tensor (on CPU).
        model (nn.Module): The sequence prediction model.
        optimizer (Optimizer): PyTorch optimizer.
        loss_fn_* (function/nn.Module): Various loss functions (Dice, GDL, L1, L2).
        args (object): Configuration arguments (must include args.device).
        batch_size (int): The number of samples to process per iteration.
        lambda_* (float): Weight multipliers for the corresponding loss terms.
        use_augmentation (bool): Flag to apply data augmentation.
        
    Returns:
        np.array: Array of total loss values recorded per batch iteration.
    """
    model.train()
    epoch_losses = []

    N_SAMPLES = len(full_clean_data_tensor_cpu)
    N_BATCHES = N_SAMPLES // batch_size
    indices = torch.randperm(N_SAMPLES).tolist()
    
    # Setup for manual iteration
    for i in tqdm(range(N_BATCHES), desc="Batch progress"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        
        data_clip = full_clean_data_tensor_cpu[batch_indices].to(args.device)

        # --- AUGMENTATION & TARGET SETUP ---
        if use_augmentation:
            # Assumes f_augment_spatial_and_intensity returns augmented input and target clips
            input_clip_for_model, target_clip_for_loss = f_augment_spatial_and_intensity(data_clip, args)
        else:
            input_clip_for_model = data_clip
            target_clip_for_loss = data_clip
        
        B, C, T_clip, H, W = input_clip_for_model.shape
        T_pred = T_clip - 1 # T_pred = 3

        optimizer.zero_grad()
        
        # --- FORWARD PASS ---
        # The model returns 9 outputs: predictions (I1..I3), raw targets, I0_hat, I0_gt_raw, and 5 feature maps
        predictions, targets_pred_clean_raw, I0_hat, I0_gt_clean_raw, E1_evolved, E2_evolved, E3_evolved, E4_evolved, E_bn_evolved = model(input_clip_for_model)

        # --- MANUALLY DERIVE CLEAN TARGETS ---
        # I0_gt_clean is the target for reconstruction (T=0)
        I0_gt_clean = target_clip_for_loss[:, :, 0, :, :] 
        # targets_pred_clean is the target for prediction (T=1, 2, 3)
        targets_pred_clean = target_clip_for_loss[:, :, 1:, :, :]

        # =======================================================
        # 1. PREDICTION LOSS TENSOR ALIGNMENT
        # =======================================================
        
        # Permute (N, C, T, H, W) to (N, T, C, H, W) then flatten to (N*T, C, H, W)
        # This treats N*T frames as a single batch for loss calculation.
        predictions_flat = predictions.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        targets_flat = targets_pred_clean.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

        # Split Tensors (Channels 0, 1, 2)
        target_faf_pred = targets_flat[:, 0:1, :, :]; out_faf_pred = predictions_flat[:, 0:1, :, :]
        target_mask_pred = targets_flat[:, 1:2, :, :]; out_mask_pred = predictions_flat[:, 1:2, :, :]
        target_residual_pred = targets_flat[:, 2:3, :, :]; out_residual_pred = predictions_flat[:, 2:3, :, :]
        
        # 1. Prediction Loss (T=1, 2, 3)
        prediction_loss = (
            (lambda_faf * loss_fn_l1(out_faf_pred, target_faf_pred)) + 
            (lambda_mask * loss_fn_dice(target_mask_pred, out_mask_pred)) + 
            (lambda_residual * loss_fn_dice(target_residual_pred, out_residual_pred)) + 
            (lambda_gdl * (loss_fn_gdl(out_faf_pred, target_faf_pred) + loss_fn_gdl(out_mask_pred, target_mask_pred) + loss_fn_gdl(out_residual_pred, target_residual_pred)))
        )

        # 2. Reconstruction Loss (T=0)
        target_faf_recon = I0_gt_clean[:, 0:1, :, :]; out_faf_recon = I0_hat[:, 0:1, :, :]
        target_mask_recon = I0_gt_clean[:, 1:2, :, :]; out_mask_recon = I0_hat[:, 1:2, :, :]
        target_residual_recon = I0_gt_clean[:, 2:3, :, :]; out_residual_recon = I0_hat[:, 2:3, :, :]
        
        reconstruction_loss = (
            (lambda_faf * loss_fn_l1(out_faf_recon, target_faf_recon)) + 
            (lambda_mask * loss_fn_dice(target_mask_recon, out_mask_recon)) + 
            (lambda_residual * loss_fn_dice(target_residual_recon, out_residual_recon)) + 
            (lambda_gdl * (loss_fn_gdl(out_faf_recon, target_faf_recon) + loss_fn_gdl(out_mask_recon, target_mask_recon) + loss_fn_gdl(out_residual_recon, target_residual_recon)))
        )
        
        # 3. Regularization Loss
        loss_bottleneck_l2 = loss_fn_l2(E_bn_evolved, torch.zeros_like(E_bn_evolved))

        # 4. Total Loss
        total_loss = (
            prediction_loss + 
            (lambda_recon * reconstruction_loss) + 
            (lambda_bottleneck * loss_bottleneck_l2)
        )

        # Backward Pass
        total_loss.backward()

        # Gradient Clipping 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 

        # Optimization Step
        optimizer.step()
        
        epoch_losses.append(total_loss.item())
        
    N_DROPPED = N_SAMPLES % batch_size
    if N_DROPPED > 0:
        print(f"Note: Dropped {N_DROPPED} samples in the final partial batch.")
            
    return np.array(epoch_losses)

# ----------------------------------------------------------------------------------------------------------------------

def calculate_total_loss(predictions, targets_pred_clean, I0_hat, I0_gt_clean, E_bn_evolved, 
                         loss_fn_l1, loss_fn_dice, loss_fn_gdl, loss_fn_l2, args, **loss_weights):
    """
    Calculates the combined loss for UPredNet: Prediction Loss + Weighted Reconstruction Loss + Regularization.
    
    Args:
        predictions (torch.Tensor): Predicted frames I1..I3 (N, C, T_pred, H, W).
        targets_pred_clean (torch.Tensor): Target frames I1..I3 (N, C, T_pred, H, W).
        I0_hat (torch.Tensor): Reconstructed frame I0 (N, C, H, W).
        I0_gt_clean (torch.Tensor): Target frame I0 (N, C, H, W).
        E_bn_evolved (torch.Tensor): Bottleneck features for regularization.
        loss_fn_* (function/nn.Module): Various loss functions.
        args (object): Configuration arguments.
        loss_weights (dict): Keyword arguments containing lambda weights.

    Returns:
        torch.Tensor: The computed total scalar loss.
    """
    B, C, T_pred, H, W = targets_pred_clean.shape
    
    # Extract loss weights from kwargs or use defaults
    lambda_gdl = loss_weights.get('lambda_gdl', 1e-3)
    lambda_faf = loss_weights.get('lambda_faf', 0.2)
    lambda_mask = loss_weights.get('lambda_mask', 1.0)
    lambda_residual = loss_weights.get('lambda_residual', 5.0)
    lambda_recon = loss_weights.get('lambda_recon', 0.1)
    lambda_bottleneck = loss_weights.get('lambda_bottleneck', 1e-6)

    # Flatten time and batch dimensions together for prediction loss: [N*T, C, H, W]
    predictions_flat = predictions.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    targets_flat = targets_pred_clean.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

    # --- 1. PREDICTION LOSS (T=1, 2, 3) ---
    target_faf_pred = targets_flat[:, 0:1, :, :]; out_faf_pred = predictions_flat[:, 0:1, :, :]
    target_mask_pred = targets_flat[:, 1:2, :, :]; out_mask_pred = predictions_flat[:, 1:2, :, :]
    target_residual_pred = targets_flat[:, 2:3, :, :]; out_residual_pred = predictions_flat[:, 2:3, :, :]
    
    prediction_loss = (
        (lambda_faf * loss_fn_l1(out_faf_pred, target_faf_pred)) + 
        (lambda_mask * loss_fn_dice(target_mask_pred, out_mask_pred)) + 
        (lambda_residual * loss_fn_dice(target_residual_pred, out_residual_pred)) + 
        (lambda_gdl * (
            loss_fn_gdl(out_faf_pred, target_faf_pred) + 
            loss_fn_gdl(out_mask_pred, target_mask_pred) + 
            loss_fn_gdl(out_residual_pred, target_residual_pred)
        ))
    )

    # --- 2. RECONSTRUCTION LOSS (T=0) ---
    target_faf_recon = I0_gt_clean[:, 0:1, :, :]; out_faf_recon = I0_hat[:, 0:1, :, :]
    target_mask_recon = I0_gt_clean[:, 1:2, :, :]; out_mask_recon = I0_hat[:, 1:2, :, :]
    target_residual_recon = I0_gt_clean[:, 2:3, :, :]; out_residual_recon = I0_hat[:, 2:3, :, :]
    
    reconstruction_loss = (
        (lambda_faf * loss_fn_l1(out_faf_recon, target_faf_recon)) + 
        (lambda_mask * loss_fn_dice(target_mask_recon, out_mask_recon)) + 
        (lambda_residual * loss_fn_dice(target_residual_recon, out_residual_recon)) + 
        (lambda_gdl * (
            loss_fn_gdl(out_faf_recon, target_faf_recon) + 
            loss_fn_gdl(out_mask_recon, target_mask_recon) + 
            loss_fn_gdl(out_residual_recon, target_residual_recon)
        ))
    )
    
    # --- 3. ARCHITECTURAL REGULARIZATION LOSSES ---
    loss_bottleneck_l2 = loss_fn_l2(E_bn_evolved, torch.zeros_like(E_bn_evolved))
    
    # --- 4. TOTAL LOSS ---
    total_loss = (
        prediction_loss + 
        (lambda_recon * reconstruction_loss) + 
        (lambda_bottleneck * loss_bottleneck_l2)
    )

    return total_loss

# ----------------------------------------------------------------------------------------------------------------------

def f_single_epoch_spatiotemporal_accumulated(full_clean_data_tensor_cpu, model, optimizer, loss_fn_bce, loss_fn_dice, loss_fn_gdl, loss_fn_l1, loss_fn_l2, args, batch_size, accumulation_steps=4, **loss_weights):
    """
    Executes a single training epoch for UPredNet model using **Gradient Accumulation**.
    The gradients are updated only after 'accumulation_steps' physical batches.
    
    Args:
        full_clean_data_tensor_cpu (torch.Tensor): The entire training dataset tensor (on CPU).
        model (nn.Module): The sequence prediction model.
        optimizer (Optimizer): PyTorch optimizer.
        loss_fn_bce (nn.Module): BCE loss (retained for signature compatibility, not explicitly used).
        loss_fn_* (function/nn.Module): Various loss functions.
        args (object): Configuration arguments (must include args.device).
        batch_size (int): The physical batch size processed per step.
        accumulation_steps (int): Number of batches to accumulate gradients over.
        loss_weights (dict): Keyword arguments containing lambda weights and optional 'use_augmentation' flag.
        
    Returns:
        np.array: Array of total loss values recorded per physical batch.
    """
    model.train()
    epoch_losses = []
    
    N_SAMPLES = len(full_clean_data_tensor_cpu)
    indices = torch.randperm(N_SAMPLES).tolist()
    
    N_PHYSICAL_BATCHES = N_SAMPLES // batch_size
    
    # Initialize gradients before the epoch starts
    optimizer.zero_grad() 
    
    for i in tqdm(range(N_PHYSICAL_BATCHES), desc="Batch progress (Accumulating)"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        data_clip = full_clean_data_tensor_cpu[batch_indices].to(args.device)

        # --- AUGMENTATION & TARGET SETUP ---
        if loss_weights.get('use_augmentation'):
            input_clip_for_model, target_clip_for_loss = f_augment_spatial_and_intensity(data_clip, args)
        else:
            input_clip_for_model = data_clip
            target_clip_for_loss = data_clip
        
        # --- FORWARD PASS ---
        # Only extract the necessary outputs for loss calculation
        predictions, _, I0_hat, _, *_, E_bn_evolved = model(input_clip_for_model)
        
        # Get Clean Targets
        I0_gt_clean = target_clip_for_loss[:, :, 0, :, :]
        targets_pred_clean = target_clip_for_loss[:, :, 1:, :, :]
        
        # Calculate total_loss using the helper function
        total_loss = calculate_total_loss(
            predictions, targets_pred_clean, I0_hat, I0_gt_clean, E_bn_evolved, 
            loss_fn_l1, loss_fn_dice, loss_fn_gdl, loss_fn_l2, args, **loss_weights
        )
        
        # Normalize loss by the accumulation factor to keep gradient magnitude correct
        normalized_loss = total_loss / accumulation_steps
        
        # --- BACKWARD PASS (Accumulate Gradients) ---
        normalized_loss.backward()

        # --- OPTIMIZER STEP (Only after K steps) ---
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad() # Reset gradients after update
            
        epoch_losses.append(total_loss.item())

    # Final step for any remaining accumulated gradients in the final partial cycle
    if N_PHYSICAL_BATCHES % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
            
    N_DROPPED = N_SAMPLES % batch_size
    if N_DROPPED > 0:
        print(f"Note: Dropped {N_DROPPED} samples in the final partial batch.")
            
    return np.array(epoch_losses)

