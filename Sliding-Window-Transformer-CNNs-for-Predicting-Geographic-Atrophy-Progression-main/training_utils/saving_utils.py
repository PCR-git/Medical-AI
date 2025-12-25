
import torch
from pathlib import Path
from datetime import datetime
import numpy as np
import os
import pickle

def save_final_experiment_data(
    model, 
    final_epoch, 
    base_save_dir, 
    k_fold_index,
    all_iteration_losses,
    epoch_iteration_counts,
    # --- Residual History ---
    train_dice_t1, train_dice_t2, train_dice_t3,
    test_dice_t1, test_dice_t2, test_dice_t3,
    train_sd_t1, train_sd_t2, train_sd_t3,
    test_sd_t1, test_sd_t2, test_sd_t3,
    # --- Mask History ---
    train_mask_t1, train_mask_t2, train_mask_t3,
    test_mask_t1, test_mask_t2, test_mask_t3,
    train_mask_sd_t1, train_mask_sd_t2, train_mask_sd_t3,
    test_mask_sd_t1, test_mask_sd_t2, test_mask_sd_t3,
    model_name_prefix
):
    """
    Saves all experiment data (model weights, dice history, loss history, metadata) 
    to a unique, timestamped directory.

    Args:
        model: The trained PyTorch model.
        final_epoch: The final epoch number.
        base_save_dir: The root directory for saving experiments.
        k_fold_index: The current fold number (k).
        [All history lists are passed directly from the training loop]
        model_name_prefix: Base name for the saved files/directory.

    Returns:
        The Path to the final experiment directory.
    """
    try:
        # 1. Define the unique folder name based on prefix and fold index
        folder_name = f"{model_name_prefix}_k_fold_{k_fold_index}"
        exp_dir = base_save_dir / folder_name

        # Ensure the directory exists
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving experiment data to: {exp_dir.absolute().as_posix()}")

        # 2. Prepare Data Dictionaries
        
        # --- Loss History ---
        loss_data = {
            'all_iteration_losses': np.array(all_iteration_losses),
            'epoch_iteration_counts': np.array(epoch_iteration_counts)
        }

        # --- Residual Dice History ---
        res_dice_history = {
            'train_t1': np.array(train_dice_t1), 'train_t2': np.array(train_dice_t2), 'train_t3': np.array(train_dice_t3),
            'test_t1': np.array(test_dice_t1), 'test_t2': np.array(test_dice_t2), 'test_t3': np.array(test_dice_t3),
            'train_sd_t1': np.array(train_sd_t1), 'train_sd_t2': np.array(train_sd_t2), 'train_sd_t3': np.array(train_sd_t3),
            'test_sd_t1': np.array(test_sd_t1), 'test_sd_t2': np.array(test_sd_t2), 'test_sd_t3': np.array(test_sd_t3)
        }

        # --- Full Mask Dice History ---
        mask_dice_history = {
            'train_t1': np.array(train_mask_t1), 'train_t2': np.array(train_mask_t2), 'train_t3': np.array(train_mask_t3),
            'test_t1': np.array(test_mask_t1), 'test_t2': np.array(test_mask_t2), 'test_t3': np.array(test_mask_t3),
            'train_sd_t1': np.array(train_mask_sd_t1), 'train_sd_t2': np.array(train_mask_sd_t3), 'train_sd_t3': np.array(train_mask_sd_t3),
            'test_sd_t1': np.array(test_mask_sd_t1), 'test_sd_t2': np.array(test_mask_sd_t2), 'test_sd_t3': np.array(test_mask_sd_t3)
        }

        # 3. Save Data (NumPy files)
        np.save(exp_dir / 'loss_history.npy', loss_data)
        np.save(exp_dir / 'residual_dice_history.npy', res_dice_history)
        np.save(exp_dir / 'mask_dice_history.npy', mask_dice_history)

        # 4. Save Model Weights (Checkpoint)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_checkpoint_epoch{final_epoch}_{timestamp}.pth"
        
        checkpoint = {
            'epoch': final_epoch,
            'model_state_dict': model.state_dict(),
            'description': f'Final weights after training on fold {k_fold_index}.',
            'model_name': model_name_prefix
        }
        torch.save(checkpoint, exp_dir / model_filename)

        print(f"Data save complete for fold {k_fold_index}.")
        return exp_dir
        
    except Exception as e:
        print(f"An error occurred during experiment data saving for fold {k_fold_index}: {e}")
        return None


    

