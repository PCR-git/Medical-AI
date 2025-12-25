
# --- EVALUATION FUNCTIONS ---

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

def f_eval_pred_dice_test_set(test_loader, model, args, soft_dice=False, use_median=True):
    model.eval()
    
    # --- Initialize Accumulators ---
    # AGGREGATE Mode Accumulators (Intersections and Sums for Channel 1 and 2, using high precision)
    total_intersections_res = [torch.tensor(0.0, dtype=torch.float64) for _ in range(3)]
    total_sum_pixels_res = [torch.tensor(0.0, dtype=torch.float64) for _ in range(3)]
    total_intersections_msk = [torch.tensor(0.0, dtype=torch.float64) for _ in range(3)]
    total_sum_pixels_msk = [torch.tensor(0.0, dtype=torch.float64) for _ in range(3)]
    
    # MEDIAN/MEAN Mode Accumulators (List to store individual scores for SD calculation)
    individual_scores_res = [[] for _ in range(3)]
    individual_scores_msk = [[] for _ in range(3)]
    
    device = args.device
    T_pred = 3
    smooth = 1e-6
    metric_name = "Median Dice" if use_median else "Aggregated Dice"
    desc_label = f"{metric_name} ({'Soft' if soft_dice else 'Hard'})"

    with torch.no_grad():
        for data_batch in tqdm(test_loader, desc=f"Evaluating Test {desc_label}"):
            predictions, targets, *discards = model(data_batch.to(device))
            
            for t in range(T_pred):
                # --- A. Extract Residual and Mask Tensors ---
                pred_res_soft = predictions[:, 2:3, t, :, :] # Channel 2: Residual
                target_res = targets[:, 2:3, t, :, :]
                
                pred_msk_soft = predictions[:, 1:2, t, :, :] # Channel 1: Mask
                target_msk = targets[:, 1:2, t, :, :]

                # --- B. Target Cleanup (Ensure target magnitude is 1.0) ---
                target_res_cleaned = (torch.abs(target_res) > 1e-4).float() 
                target_msk_cleaned = (torch.abs(target_msk) > 1e-4).float() 
                
                # --- C. Prediction Selection ---
                if soft_dice:
                    pred_res_tensor, pred_msk_tensor = pred_res_soft, pred_msk_soft
                else:
                    pred_res_tensor = (pred_res_soft > 0.1).float() # Optimized 0.1 threshold for Residual
                    pred_msk_tensor = (pred_msk_soft > 0.5).float() # Standard 0.5 threshold for Mask
                
                # Flatten Tensors
                pred_res_f = torch.flatten(pred_res_tensor); target_res_f = torch.flatten(target_res_cleaned)
                pred_msk_f = torch.flatten(pred_msk_tensor); target_msk_f = torch.flatten(target_msk_cleaned)
                
                # --- D. ACCUMULATION / MEDIAN BRANCH ---
                if use_median:
                    # Calculate Dice for the current batch (B items)
                    score_res = (2.0 * torch.sum(target_res_f * pred_res_f)) / (torch.sum(target_res_f) + torch.sum(pred_res_f) + smooth)
                    score_msk = (2.0 * torch.sum(target_msk_f * pred_msk_f)) / (torch.sum(target_msk_f) + torch.sum(pred_msk_f) + smooth)
                    
                    individual_scores_res[t].append(score_res.item())
                    individual_scores_msk[t].append(score_msk.item())
                else:
                    # Accumulate sums for Aggregated Dice (using high-precision accumulators)
                    total_intersections_res[t] += torch.sum(target_res_f * pred_res_f).item()
                    total_sum_pixels_res[t] += (torch.sum(target_res_f) + torch.sum(pred_res_f)).item()
                    total_intersections_msk[t] += torch.sum(target_msk_f * pred_msk_f).item()
                    total_sum_pixels_msk[t] += (torch.sum(target_msk_f) + torch.sum(pred_msk_f)).item()

    # --- Final Score Calculation and SD ---
    if use_median:
        # Calculate Median and SD for Residuals
        res_scores = [np.median(scores) for scores in individual_scores_res]
        res_sds = [np.std(scores) for scores in individual_scores_res]
        # Calculate Median and SD for Masks
        msk_scores = [np.median(scores) for scores in individual_scores_msk]
        msk_sds = [np.std(scores) for scores in individual_scores_msk]
    else:
        # Aggregated Dice calculation (no SD is possible for a single aggregate score)
        res_scores = [(2.0 * total_intersections_res[t].item()) / (total_sum_pixels_res[t].item() + smooth) for t in range(T_pred)]
        msk_scores = [(2.0 * total_intersections_msk[t].item()) / (total_sum_pixels_msk[t].item() + smooth) for t in range(T_pred)]
        res_sds = [0.0] * T_pred # Set SDs to zero for plotting consistency
        msk_sds = [0.0] * T_pred
    
    model.train()
    # RETURN: ((res_scores, res_sds), (msk_scores, msk_sds))
    return (res_scores, res_sds), (msk_scores, msk_sds)


def f_eval_pred_dice_train_set(full_data_tensor_cpu, model, args, batch_size, soft_dice=False, use_median=True):
    model.eval()
    
    # 1. Initialize Accumulators
    total_intersections_res = [torch.tensor(0.0, dtype=torch.float64) for _ in range(3)]
    total_sum_pixels_res = [torch.tensor(0.0, dtype=torch.float64) for _ in range(3)]
    total_intersections_msk = [torch.tensor(0.0, dtype=torch.float64) for _ in range(3)]
    total_sum_pixels_msk = [torch.tensor(0.0, dtype=torch.float64) for _ in range(3)] 
    
    individual_scores_res = [[] for _ in range(3)]
    individual_scores_msk = [[] for _ in range(3)]
    
    device = args.device
    T_pred = 3
    smooth = 1e-6

    N_BATCHES = len(full_data_tensor_cpu) // batch_size
    full_clean_data = full_data_tensor_cpu
    
    metric_name = "Median Dice" if use_median else "Aggregated Dice"
    desc_label = f"{metric_name} ({'Soft' if soft_dice else 'Hard'})"

    with torch.no_grad():
        for i in tqdm(range(N_BATCHES), desc=f"Evaluating Train {desc_label}"):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            data_batch = full_clean_data[start_idx:end_idx].to(device)

            predictions, targets, *discards = model(data_batch)
            
            for t in range(T_pred):
                # --- A. Extract Residual and Mask Tensors ---
                pred_res_soft = predictions[:, 2:3, t, :, :]
                target_res = targets[:, 2:3, t, :, :]
                
                pred_msk_soft = predictions[:, 1:2, t, :, :]
                target_msk = targets[:, 1:2, t, :, :]

                # --- B. Target Cleanup ---
                target_res_cleaned = (torch.abs(target_res) > 1e-4).float() 
                target_msk_cleaned = (torch.abs(target_msk) > 1e-4).float() 
                
                # --- C. Prediction Selection ---
                if soft_dice:
                    pred_res_tensor, pred_msk_tensor = pred_res_soft, pred_msk_soft
                else:
                    pred_res_tensor = (pred_res_soft > 0.1).float()
                    pred_msk_tensor = (pred_msk_soft > 0.5).float()
                
                pred_res_f = torch.flatten(pred_res_tensor); target_res_f = torch.flatten(target_res_cleaned)
                pred_msk_f = torch.flatten(pred_msk_tensor); target_msk_f = torch.flatten(target_msk_cleaned)

                # --- D. ACCUMULATION / MEDIAN BRANCH ---
                if use_median:
                    score_res = (2.0 * torch.sum(target_res_f * pred_res_f)) / (torch.sum(target_res_f) + torch.sum(pred_res_f) + smooth)
                    score_msk = (2.0 * torch.sum(target_msk_f * pred_msk_f)) / (torch.sum(target_msk_f) + torch.sum(pred_msk_f) + smooth)
                    
                    individual_scores_res[t].append(score_res.item())
                    individual_scores_msk[t].append(score_msk.item())
                else:
                    # Accumulate Aggregated Dice
                    total_intersections_res[t] += torch.sum(target_res_f * pred_res_f).item()
                    total_sum_pixels_res[t] += (torch.sum(target_res_f) + torch.sum(pred_res_f)).item()
                    total_intersections_msk[t] += torch.sum(target_msk_f * pred_msk_f).item()
                    total_sum_pixels_msk[t] += (torch.sum(target_msk_f) + torch.sum(pred_msk_f)).item()
            
    # --- Final Score Calculation ---
    if use_median:
        res_scores = [np.median(scores) for scores in individual_scores_res]
        res_sds = [np.std(scores) for scores in individual_scores_res]
        msk_scores = [np.median(scores) for scores in individual_scores_msk]
        msk_sds = [np.std(scores) for scores in individual_scores_msk]
    else:
        # Aggregated Dice calculation
        res_scores = [(2.0 * total_intersections_res[t].item()) / (total_sum_pixels_res[t].item() + smooth) for t in range(T_pred)]
        msk_scores = [(2.0 * total_intersections_msk[t].item()) / (total_sum_pixels_msk[t].item() + smooth) for t in range(T_pred)]
        res_sds = [0.0] * T_pred
        msk_sds = [0.0] * T_pred
    
    model.train()
    return (res_scores, res_sds), (msk_scores, msk_sds)


# --- PLOTTING AND VISUALIZATION FUNCTIONS ---
    
def plot_train_test_dice_history(train_dice_t1, train_dice_t2, train_dice_t3, 
                                 test_dice_t1, test_dice_t2, test_dice_t3,
                                 train_sd_t1, train_sd_t2, train_sd_t3, 
                                 test_sd_t1, test_sd_t2, test_sd_t3,
                                 plot_title='Dice Score History'):
    """
    Plots the historical trend of the Dice Score (Mean or Median) and its Standard Deviation (Error Bars)
    across epochs for training and testing sets, separated by prediction time step (T=1, T=2, T=3).
    
    Args:
        train_dice/test_dice (list): Lists of mean or median Dice scores per epoch.
        train_sd/test_sd (list): Lists of corresponding Standard Deviation values per epoch.
        plot_title (str): Custom title for the plot.
    """

    plt.figure(figsize=(12, 8))
    
    epochs = np.arange(1, len(train_dice_t1) + 1) if train_dice_t1 else []
    
    # Data structure for iterating and plotting
    plot_data = [
        # Training Data (T=1, T=2, T=3)
        (train_dice_t1, train_sd_t1, 'skyblue', 'Train T=1', 'o', '-'),
        (train_dice_t2, train_sd_t2, 'blue', 'Train T=2', '^', '--'),
        (train_dice_t3, train_sd_t3, 'darkblue', 'Train T=3 (Critical)', 's', '-'),
        # Testing Data (T=1, T=2, T=3)
        (test_dice_t1, test_sd_t1, 'lightcoral', 'Test T=1', 'o', '-'),
        (test_dice_t2, test_sd_t2, 'red', 'Test T=2', '^', '--'),
        (test_dice_t3, test_sd_t3, 'darkred', 'Test T=3 (Critical)', 's', '-'),
    ]

    for scores, sds, color, label, marker, linestyle in plot_data:
        if scores:
            # Plot scores using error bars to visualize standard deviation (SD)
            plt.errorbar(epochs, scores, yerr=sds,
                         label=label,
                         color=color,
                         marker=marker, 
                         linestyle=linestyle,
                         capsize=3,         # Size of the caps on the error bars
                         elinewidth=1.5,    # Thickness of the error lines
                         linewidth=2,
                         alpha=0.8)

    plt.title(plot_title)
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score ± SD')
    plt.ylim(0, 1.05)  # Set y-axis limit for Dice Score range
    plt.grid(True, linestyle='--')
    plt.legend(loc='lower right', fontsize='small')
    plt.show()
    
    
def soft_dice_score(y_true_f, y_pred_f, smooth=1e-6):
    """
    Calculates the Soft Dice Coefficient using squared terms in the denominator 
    for improved numerical stability on sparse targets.
    
    y_true_f: Flattened target (binary)
    y_pred_f: Flattened prediction (soft or hard)
    """
    
    # 1. Intersection (y * y_hat)
    intersection = torch.sum(y_true_f * y_pred_f)
    
    # 2. Denominator using squared terms (L2 norm)
    # y_true_f is binary, so sum(y_true_f^2) == sum(y_true_f)
    # y_pred_f is soft/float, so sum(y_pred_f^2) is essential.
    denominator = torch.sum(y_true_f * y_true_f) + torch.sum(y_pred_f * y_pred_f)
    
    # Final Score
    score = (2. * intersection + smooth) / (denominator + smooth)
    
    return score


def f_get_individual_dice(dataset, model, args, is_train_set, soft_dice=True, hard_dice_threshold=0.1):
    """
    Calculates the Dice score (Hard or Soft) for the Residual Mask (Ch 2)
    and the Mask (Ch 1) for every single clip (batch_size=1).
    
    Returns: 
        (residual_scores_array, mask_scores_array), 
        (res_mean_sd, msk_mean_sd) -- tuple of summary metrics
    """
    model.eval()
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    all_clip_dice_scores_res = []
    all_clip_dice_scores_msk = []
    
    device = args.device
    T_pred = 3
    desc_label = "Soft Dice (Individual)" if soft_dice else "Hard Dice (Individual)"

    with torch.no_grad():
        if is_train_set:
            iterator = tqdm(data_loader, desc=f"Evaluating Train Clips ({desc_label})")
        else:
            iterator = tqdm(data_loader, desc=f"Evaluating Test Clips ({desc_label})")
            
        for data_batch in iterator:
            predictions, targets, *discards = model(data_batch.to(device))
            
            current_clip_scores_res = []
            current_clip_scores_msk = []
            
            for t in range(T_pred):
                # --- A. Extraction and Cleanup ---
                pred_res_soft, target_res = predictions[:, 2:3, t, :, :], targets[:, 2:3, t, :, :]
                pred_msk_soft, target_msk = predictions[:, 1:2, t, :, :], targets[:, 1:2, t, :, :]
                
                target_res_cleaned = (torch.abs(target_res) > 1e-4).float()
                target_msk_cleaned = (torch.abs(target_msk) > 1e-4).float()
                
                # --- B. Prediction Selection ---
                if soft_dice:
                    pred_res_f, pred_msk_f = torch.flatten(pred_res_soft), torch.flatten(pred_msk_soft)
                else:
                    pred_res_f = torch.flatten((pred_res_soft > hard_dice_threshold).float())
                    pred_msk_f = torch.flatten((pred_msk_soft > 0.5).float())

                # --- C. Score Calculation ---
                target_res_f, target_msk_f = torch.flatten(target_res_cleaned), torch.flatten(target_msk_cleaned)

                score_res = soft_dice_score(target_res_f, pred_res_f).item()
                score_msk = soft_dice_score(target_msk_f, pred_msk_f).item()
                
                current_clip_scores_res.append(score_res)
                current_clip_scores_msk.append(score_msk)
            
            all_clip_dice_scores_res.append(current_clip_scores_res)
            all_clip_dice_scores_msk.append(current_clip_scores_msk)
            
    model.train()
    
    # Convert to NumPy arrays for easy statistical operations
    res_array = np.array(all_clip_dice_scores_res)
    msk_array = np.array(all_clip_dice_scores_msk)
    
    # --- Calculate Summary Statistics (Mean and SD for all 3 time steps) ---
    res_mean_sd = [(np.mean(res_array[:, t]), np.std(res_array[:, t])) for t in range(T_pred)]
    msk_mean_sd = [(np.mean(msk_array[:, t]), np.std(msk_array[:, t])) for t in range(T_pred)]
    
    # Return the individual scores (for plotting) AND the summary statistics
    return (res_array, msk_array), (res_mean_sd, msk_mean_sd)


def f_plot_individual_dice(individual_scores_train, individual_scores_test, metric_type, channel_name='Residual Mask'):
    """
    Plots the individual Dice scores for every clip for T=1, T=2, and T=3 predictions 
    for both the training and testing sets, providing aggregate statistics in the title.
    
    Args:
        individual_scores_train (list or np.array): List of [T1, T2, T3] scores for each clip in the train set.
        individual_scores_test (list or np.array): List of [T1, T2, T3] scores for each clip in the test set.
        metric_type (str): The metric being plotted (e.g., 'Dice Score', 'Jaccard Index').
        channel_name (str): The name of the segmentation channel being evaluated.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    
    scores_train = np.array(individual_scores_train)
    scores_test = np.array(individual_scores_test)

    # Calculate Summary Statistics for each time step (T=1, T=2, T=3)
    train_mean = np.mean(scores_train, axis=0) 
    train_std = np.std(scores_train, axis=0)   
    test_mean = np.mean(scores_test, axis=0)
    test_std = np.std(scores_test, axis=0)

    # --- Plot 1: Training Set ---
    ax1 = axes[0]
    clip_indices_train = np.arange(len(scores_train))

    ax1.plot(clip_indices_train, scores_train[:, 0], label=f'T=1 Mean: {train_mean[0]:.3f} (SD: {train_std[0]:.3f})', linestyle='-', marker='.', alpha=0.6)
    ax1.plot(clip_indices_train, scores_train[:, 1], label=f'T=2 Mean: {train_mean[1]:.3f} (SD: {train_std[1]:.3f})', linestyle='--', marker='.', alpha=0.6)
    ax1.plot(clip_indices_train, scores_train[:, 2], label=f'T=3 Mean: {train_mean[2]:.3f} (SD: {train_std[2]:.3f})', linestyle='-', marker='o', linewidth=2)
    
    # Set plot title including overall T=3 performance
    title_train = f'Train Set: {channel_name} {metric_type} (N={len(scores_train)})'
    title_train += f'\nOverall Mean T3: {train_mean[2]:.3f} (SD: {train_std[2]:.3f})'
    ax1.set_title(title_train, fontsize=10)
    
    ax1.set_xlabel('Clip Index')
    ax1.set_ylabel(f'{metric_type} Score')
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', fontsize='small')


    # --- Plot 2: Testing Set ---
    ax2 = axes[1]
    clip_indices_test = np.arange(len(scores_test))

    ax2.plot(clip_indices_test, scores_test[:, 0], label=f'T=1 Mean: {test_mean[0]:.3f} (SD: {test_std[0]:.3f})', linestyle='-', marker='.', alpha=0.6)
    ax2.plot(clip_indices_test, scores_test[:, 1], label=f'T=2 Mean: {test_mean[1]:.3f} (SD: {test_std[1]:.3f})', linestyle='--', marker='.', alpha=0.6)
    ax2.plot(clip_indices_test, scores_test[:, 2], label=f'T=3 Mean: {test_mean[2]:.3f} (SD: {test_std[2]:.3f})', linestyle='-', marker='o', linewidth=2)
    
    # Set plot title including overall T=3 performance
    title_test = f'Test Set: {channel_name} {metric_type} (N={len(scores_test)})'
    title_test += f'\nOverall Mean T3: {test_mean[2]:.3f} (SD: {test_std[2]:.3f})'
    ax2.set_title(title_test, fontsize=10)
    
    ax2.set_xlabel('Clip Index')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', fontsize='small')

    fig.suptitle(f"Individual Clip Performance on {channel_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

