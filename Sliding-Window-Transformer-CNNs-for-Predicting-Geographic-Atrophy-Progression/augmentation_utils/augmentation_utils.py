import torch

# ==============================================================================
# PER-BATCH AUGMENTATION
# ==============================================================================

def f_augment_spatial_and_intensity(clean_data_batch, args, flip_prob=0.5):
    """
    1. Applies spatial transforms (rot/flip) to the clean data (produces the target base).
    2. Applies intensity noise/distortion based on a continuous random multiplier 
        to the FAF channel (Ch 0) of a cloned tensor (produces the noisy input).
    
    Returns: 
      - noisy_input_batch: [B, C, T, H, W] with spatially aligned, corrupted FAF.
      - clean_spatial_target_batch: [B, C, T, H, W] with spatially aligned, CLEAN FAF.
    """
    # 1. Apply spatial transformation to a clone, move to GPU
    spatial_transformed_data = clean_data_batch.clone().to(args.device)
    B, C, T, H, W = spatial_transformed_data.shape
    
    # --- 1a. Global (Spatial) Transforms (Applied to all B*T frames consistently) ---
    # NOTE: Dims=(-2, -1) assumes (B, C, T, H, W)
    if torch.rand(1).item() < 0.25: 
        spatial_transformed_data = torch.rot90(spatial_transformed_data, k=1, dims=(-2, -1))
    if torch.rand(1).item() < flip_prob:
        spatial_transformed_data = torch.flip(spatial_transformed_data, dims=[-1])
    if torch.rand(1).item() < flip_prob:
        spatial_transformed_data = torch.flip(spatial_transformed_data, dims=[-2])
        
    # This is the ideal target: Spatially aligned but FAF is still clean.
    clean_spatial_target_batch = spatial_transformed_data 
    
    # 2. Create the Noisy Input by cloning the target and corrupting FAF (Ch 0)
    noisy_input_batch = clean_spatial_target_batch.clone() 
    faf_channel_noisy = noisy_input_batch[:, 0:1, :, :, :] # [B, 1, T, H, W]
    
    # --- 2a. Continuous Brightness/Contrast Jitter & Gamma Correction ---
    
    # Draw a single intensity multiplier M ~ U(0.0, 1.0) for each sample: [B, 1, 1, 1, 1]
    intensity_multiplier = torch.rand(B, 1, 1, 1, 1, device=args.device) 

    # Define maximum augmentation bounds
    BC_MAX_JITTER = 0.4      # Max factor of +/- 20% from 1.0 (Range 0.8 to 1.2)
    GAMMA_MAX_RANGE = 0.6    # Max range 0.7 to 1.3 (Offset is 0.7)
    NOISE_MAX_SIGMA = 0.05
    
    # Scale ranges by the multiplier for each sample
    bc_range_scaled = intensity_multiplier * BC_MAX_JITTER
    gamma_range_scaled = intensity_multiplier * GAMMA_MAX_RANGE
    
    # Draw unique factors for each sample/time step: Shape [B, 1, T, 1, 1]
    shape_BT = (B, 1, T, 1, 1)

    # Calculate actual B/C factors
    brightness_factor = torch.rand(shape_BT, device=args.device) * bc_range_scaled + (1.0 - bc_range_scaled / 2)
    contrast_factor = torch.rand(shape_BT, device=args.device) * bc_range_scaled + (1.0 - bc_range_scaled / 2)
    
    # Calculate actual Gamma factors
    gamma_offset = 0.7 
    gamma_factor = torch.rand(shape_BT, device=args.device) * gamma_range_scaled + gamma_offset

    # Apply B/C/Gamma
    faf_channel_noisy = faf_channel_noisy * brightness_factor
    mean_val = 0.5
    faf_channel_noisy = (faf_channel_noisy - mean_val) * contrast_factor + mean_val
    faf_channel_noisy = torch.pow(faf_channel_noisy.abs() + 1e-6, gamma_factor)

    # --- 2b. Gaussian Noise (Additive) ---
    # Scale sigma by the multiplier drawn for this sample: [B, 1, 1, 1, 1]
    noise_sigma_scaled = intensity_multiplier * NOISE_MAX_SIGMA
    
    noise = torch.randn_like(faf_channel_noisy) * noise_sigma_scaled
    faf_channel_noisy = faf_channel_noisy + noise

    # --- 3. Final Clipping and Assignment ---
    epsilon_clamp_data = 1e-4
    noisy_input_batch[:, 0:1, :, :, :] = torch.clamp(faf_channel_noisy, epsilon_clamp_data, 1.0 - epsilon_clamp_data)
    
    return noisy_input_batch.contiguous(), clean_spatial_target_batch.contiguous()



