#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt

def f_display_autoencoder(dataset, model, args, sample_idx=0, time_step=0):
    """
    Displays the Autoencoder's 3-channel input (FAF, Mask, Residual) 
    and its 3-channel reconstruction in a 2x3 grid for a single time step.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset (e.g., train_dataset or test_dataset) 
            which supports __getitem__ indexing.
        model (torch.nn.Module): The Autoencoder model to evaluate.
        args (object): Object containing configuration arguments, including args.device.
        sample_idx (int): Index of the clip in the dataset to visualize.
        time_step (int): Time index (T) of the frame within the clip (0-3).
    """
    model.eval() # Set model to evaluation mode for inference
    
    # 1. Retrieve the single data sample
    if len(dataset) <= sample_idx:
        print(f"Dataset too small for visualization index {sample_idx}. Skipping display.")
        return

    # data_item shape: [C=3, T=4, H=256, W=256]
    data_item = dataset[sample_idx]
    
    # 2. Extract the full 3-channel input frame at the specified time step
    # input_frame_3ch shape: [C=3, H=256, W=256]
    input_frame_3ch = data_item[:, time_step, :, :] 
    
    # 3. Prepare input for the model
    # Model input shape: [B=1, C=3, H=256, W=256]
    model_input = input_frame_3ch.unsqueeze(0).to(args.device) 
    
    with torch.no_grad():
        # Perform forward pass (assuming the model returns multiple outputs, 
        # but only the first (reconstructed frames) is used here).
        # out_frames shape: [1, 3, 256, 256]
        out_frames, _, _, _, _, _ = model(model_input)
    
    # 4. Convert input and output tensors to numpy for plotting
    # Shapes are converted to (C, H, W) for plotting access
    input_np = input_frame_3ch.cpu().numpy()
    output_np = out_frames.squeeze().cpu().numpy()

    # 5. Plotting in a 2x3 grid (Rows: Ground Truth, Reconstruction; Columns: FAF, Mask, Residual)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12)) 
    
    channel_names = ["FAF (Ch 0)", "Mask (Ch 1)", "Residual (Ch 2)"]
    
    fig.suptitle(
        f'Autoencoder Performance (Sample {sample_idx}, Time Step {time_step}): Ground Truth vs. Reconstruction', 
        fontsize=16, y=1.02
    )

    # Iterate through the 3 channels
    for i in range(3):
        # Row 1: Ground Truth
        ax_gt = axes[0, i]
        ax_gt.imshow(input_np[i], cmap='gray')
        ax_gt.set_title(f"Ground Truth: {channel_names[i]}")
        ax_gt.axis('off')

        # Row 2: Reconstruction
        ax_rec = axes[1, i]
        ax_rec.imshow(output_np[i], cmap='gray')
        ax_rec.set_title(f"Reconstruction: {channel_names[i]}")
        ax_rec.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    model.train() # Restore model to training mode

def plot_log_loss(all_iteration_losses, epoch_list):
    """ 
    Plots the logarithm of the total loss across all training iterations.
    Vertical lines indicate the transition between epochs.
    
    Args:
        all_iteration_losses (list): List of total loss values recorded after each training iteration.
        epoch_list (list): List where each element is the number of iterations in the corresponding epoch.
    """
    
    if not all_iteration_losses:
        print("No loss data to plot.")
        return
        
    all_losses = np.array(all_iteration_losses)
    # Apply log transformation, clipping values near zero to prevent log(0)
    log_losses = np.log(np.maximum(all_losses, 1e-6))
    
    plt.figure(figsize=(10, 6))
    plt.plot(log_losses, label='Total Log Loss per Iteration', color='blue')
    
    # Calculate cumulative iteration indices for epoch markers
    cumulative_iters = 0
    epoch_markers = []
    for num_iters in epoch_list:
        cumulative_iters += num_iters
        epoch_markers.append(cumulative_iters)
    
    # Draw vertical lines to separate epochs
    for marker in epoch_markers[:-1]:
        plt.axvline(x=marker, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
    plt.title('Total Log Loss Over Training Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Log(Total Loss)')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.show()
    

# def f_display_frames(dataset, model, args, sample_idx=0, T_total=4):
#     """
#     Displays the Ground Truth and model's output sequence (reconstruction of I0, 
#     prediction of I1, I2, I3) across all 4 time steps (T_total=4) for all 3 channels.
    
#     Args:
#         dataset (torch.utils.data.Dataset): The clip dataset.
#         model (torch.nn.Module): The sequence prediction model.
#         args (object): Object containing configuration arguments, including args.device.
#         sample_idx (int): Index of the clip in the dataset to visualize.
#         T_total (int): Total number of frames in the sequence (must be 4).
#     """
#     model.eval()  # Set model to evaluation mode
    
#     if len(dataset) <= sample_idx:
#         print(f"Dataset too small for visualization index {sample_idx}. Skipping display.")
#         return

#     # Data item shape: [C=3, T=4, H, W] 
#     data_item = dataset[sample_idx]  
    
#     # 1. Prepare Model Input 
#     # in_clip shape: [B=1, C=3, T=4, H, W] 
#     in_clip = data_item.unsqueeze(0).to(args.device) 
    
#     with torch.no_grad():
#         # Forward Pass: unpacks model outputs.
#         # predictions (I1..I3): [B, C, T_pred=3, H, W]
#         # I0_hat (Reconstruction of I0): [B, C, H, W]
#         predictions, _, I0_hat, _, *discards = model(in_clip)
    
#     # --- Data Preparation ---
    
#     # 1. Ground Truth Sequence (I0, I1, I2, I3)
#     # Reshape: [T=4, H, W, C=3] for Matplotlib plotting
#     gt_sequence_np = in_clip.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()

#     # 2. Predicted/Reconstructed Sequence (I_hat_0, I_hat_1, I_hat_2, I_hat_3)
#     # Unsqueeze time dimension for the reconstructed I0 frame
#     I0_hat_time = I0_hat.unsqueeze(2).cpu() # [1, C, 1, H, W]
    
#     # Concatenate the reconstructed I0 with the predicted sequence I1-I3
#     pred_sequence = torch.cat([I0_hat_time, predictions.cpu()], dim=2) # [1, C, 4, H, W]
    
#     # Final reshape: [T=4, H, W, C=3]
#     pred_sequence_np = pred_sequence.squeeze(0).permute(1, 2, 3, 0).numpy()
    
#     # --- Plotting ---
#     # Create a 6x4 grid (3 channels * 2 rows [GT/OUT] x 4 time steps)
#     fig, axes = plt.subplots(6, T_total, figsize=(16, 16))
#     plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
#     channel_names = ["FAF (Ch 0)", "Mask (Ch 1)", "Residual (Ch 2)"]

#     for i in range(in_clip.size(1)): # Iterate over channels (C=3)
#         channel_title = channel_names[i]
#         gt_row = 2 * i      # GT rows: 0, 2, 4
#         out_row = 2 * i + 1 # Output rows: 1, 3, 5

#         # Set Row Titles (Y-axis labels)
#         axes[gt_row, 0].set_ylabel(f'GT: {channel_title}', rotation=90, labelpad=15, fontsize=15)
#         axes[out_row, 0].set_ylabel(f'OUT: {channel_title}', rotation=90, labelpad=15, fontsize=15)
        
#         for t in range(T_total): # Iterate over time (0, 1, 2, 3)
#             # Get Data for the Current Time Step and Channel
#             gt_frame = gt_sequence_np[t, :, :, i]
#             output_frame = pred_sequence_np[t, :, :, i]
            
#             # Determine column title
#             if t == 0:
#                 col_title = f'Time {t} (Recon)' # Frame 0 is a reconstruction
#             else: 
#                 col_title = f'Time {t} (Prediction)' # Frames 1-3 are predictions

#             # B. Plot Ground Truth 
#             ax_gt = axes[gt_row, t]
#             ax_gt.imshow(gt_frame, cmap='gray', vmin=0, vmax=1)
#             ax_gt.set_xticks([]); ax_gt.set_yticks([])
#             if gt_row == 0:
#                  ax_gt.set_title(col_title, fontsize=10)
            
#             # C. Plot Output 
#             ax_out = axes[out_row, t]
#             ax_out.imshow(output_frame, cmap='gray', vmin=0, vmax=1)
#             ax_out.set_xticks([]); ax_out.set_yticks([])

#     fig.suptitle(f'Video Prediction Performance (I0 Recon, I1-3 Prediction) (Sample {sample_idx})', fontsize=18, y=1.01)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.98])
#     plt.show()

#     model.train() # Restore model to training mode

def f_display_frames(dataset, model, args, sample_idx=0, T_total=4):
    """
    Displays the Ground Truth and model's output sequence (reconstruction of I0, 
    prediction of I1, I2, I3) across all 4 time steps (T_total=4) for all 3 channels.
    
    Args:
        dataset (torch.utils.data.Dataset): The clip dataset.
        model (torch.nn.Module): The sequence prediction model.
        args (object): Object containing configuration arguments, including args.device.
        sample_idx (int): Index of the clip in the dataset to visualize.
        T_total (int): Total number of frames in the sequence (must be 4).
    """
    model.eval()  # Set model to evaluation mode
    
    if len(dataset) <= sample_idx:
        print(f"Dataset too small for visualization index {sample_idx}. Skipping display.")
        return

    # Data item shape: [C=3, T=4, H, W] 
    data_item = dataset[sample_idx]  
    
    # 1. Prepare Model Input 
    # in_clip shape: [B=1, C=3, T=4, H, W] 
    in_clip = data_item.unsqueeze(0).to(args.device) 
    
    with torch.no_grad():
        # Forward Pass
        predictions, _, I0_hat, _, *discards = model(in_clip)
    
    # --- Data Preparation ---
    gt_sequence_np = in_clip.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    I0_hat_time = I0_hat.unsqueeze(2).cpu() 
    pred_sequence = torch.cat([I0_hat_time, predictions.cpu()], dim=2) 
    pred_sequence_np = pred_sequence.squeeze(0).permute(1, 2, 3, 0).numpy()
    
    # --- Plotting ---
    # Width reduced to 10 to force images closer together horizontally.
    fig, axes = plt.subplots(6, T_total, figsize=(10, 14), 
                             gridspec_kw={'hspace': 0.01, 'wspace': 0.1})
    
    # Adjust margins: left must be large enough for the Y-labels
    plt.subplots_adjust(left=0.15, right=0.98, top=0.94, bottom=0.02)
    
    channel_names = ["FAF (Ch 0)", "Mask (Ch 1)", "Residual (Ch 2)"]

    for i in range(in_clip.size(1)): 
        channel_title = channel_names[i]
        gt_row = 2 * i      
        out_row = 2 * i + 1 

        # Set Row Titles
        axes[gt_row, 0].set_ylabel(f'GT: {channel_title}', rotation=90, labelpad=10, fontsize=10)
        axes[out_row, 0].set_ylabel(f'OUT: {channel_title}', rotation=90, labelpad=10, fontsize=10)
        
        for t in range(T_total): 
            gt_frame = gt_sequence_np[t, :, :, i]
            output_frame = pred_sequence_np[t, :, :, i]
            
            if t == 0:
                col_title = f'T{t} (Rec)' 
            else: 
                col_title = f'T{t} (Pred)'

            # Plot Ground Truth 
            ax_gt = axes[gt_row, t]
            ax_gt.imshow(gt_frame, cmap='gray', vmin=0, vmax=1)
            ax_gt.set_xticks([]); ax_gt.set_yticks([])
            if gt_row == 0:
                 ax_gt.set_title(col_title, fontsize=9, pad=2)
            
            # Plot Output 
            ax_out = axes[out_row, t]
#             ax_out.imshow(output_frame, cmap='gray', vmin=0, vmax=1)
            ax_out.imshow(output_frame>0.5, cmap='gray', vmin=0, vmax=1)
            ax_out.set_xticks([]); ax_out.set_yticks([])

    fig.suptitle(f'Video Prediction Performance (Sample {sample_idx})', fontsize=14, y=0.98)
    
    # No tight_layout to ensure wspace/hspace are respected
    plt.show()

    model.train()