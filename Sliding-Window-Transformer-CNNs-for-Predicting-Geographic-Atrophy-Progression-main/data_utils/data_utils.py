import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class DataWrapper(Dataset):
    """
    Handles data indexing, feature combination, and final preprocessing 
    efficiently for a single sample.
    
    The output shape per sample is [3, 4, 256, 256], representing (Channels, Time, Height, Width).
    The channels correspond to FAF (0), Mask (1), and Residual (2).
    """
    def __init__(self, fafs, masks, residuals):
        """
        Initializes the DataWrapper with input tensors for FAF images, masks, and residual masks.

        Args:
            fafs (torch.Tensor): Tensor containing FAF images.
            masks (torch.Tensor): Tensor containing general masks.
            residuals (torch.Tensor): Tensor containing residual/growth masks.
        """
        self.fafs = fafs
        self.masks = masks
        self.residuals = residuals
        assert len(self.fafs) == len(self.masks) == len(self.residuals), "Input tensors must have matching length."

    def __len__(self):
        """Returns the total number of samples (clips) in the dataset."""
        return len(self.fafs)

    def __getitem__(self, idx):
        """
        Retrieves a single sample by index, combines features, and applies final preprocessing.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            torch.Tensor: The combined and processed data tensor with shape [3, 4, 256, 256].
        """
        sample_faf = self.fafs[idx]
        sample_mask = self.masks[idx]
        sample_residual = self.residuals[idx]

        # Concatenate features along the CHANNEL dimension (dim=0).
        # Resulting shape before processing: [3, 4, 256, 256]
        combined_data = torch.cat(
            (sample_faf, sample_mask, sample_residual), 
            dim=0
        )

        # Apply a sharpness filter/binarization to the Mask (Ch 1) and Residual (Ch 2) regions.
        # Pixels with values > 1e-3 are set to 1.0 (float), enforcing binary behavior.
        combined_data[1:, :, :, :] = (combined_data[1:, :, :, :] > 1e-3).float()
        
        return combined_data
    

# def visualize_sample(train_dataset, test_dataset, sample_idx, dataset_name='test'):
#     """
#     Visualizes the three channels (FAF, Mask, Residual) for all 4 time steps 
#     of a single sample from the specified dataset (train or test).

#     Args:
#         train_dataset (Dataset): The training dataset.
#         test_dataset (Dataset): The testing dataset.
#         sample_idx (int): Index of the clip in the dataset to visualize.
#         dataset_name (str): Specifies which dataset to use ('train' or 'test').
#     """
    
#     if dataset_name == 'train':
#         dataset = train_dataset
#     elif dataset_name == 'test':
#         dataset = test_dataset
#     else:
#         print(f"Error: Invalid dataset_name '{dataset_name}'. Must be 'train' or 'test'.")
#         return

#     print(f"\n--- Visualization Example (Sample Index {sample_idx} from {dataset_name} Dataset) ---")

#     if len(dataset) <= sample_idx:
#         print(f"{dataset_name.capitalize()} dataset too small for visualization index {sample_idx}. Skipping visualization.")
#         return

#     # Data_item size: [3, 4, 256, 256] (Channels, Time, H, W)
#     Data_item = dataset[sample_idx]

#     # Visualization loop: Iterate through all 4 time steps (j=0 to 3)
#     for j in np.arange(4):
#         # Extract data for the current time step (j) and convert to numpy for plotting
#         faf = Data_item[0, j, :, :].cpu().detach().numpy()  # Channel 0
#         mask = Data_item[1, j, :, :].cpu().detach().numpy() # Channel 1
#         residual = Data_item[2, j, :, :].cpu().detach().numpy() # Channel 2

#         # Create a single figure with three subplots (1 row, 3 columns)
#         fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#         fig.suptitle(f'Time Step {j} - Data Visualization (Sample Index {sample_idx}, Set: {dataset_name.upper()})', fontsize=16)

#         # Plot 1: FAF (Channel 0)
#         ax1 = axes[0]
#         ax1.imshow(faf, cmap='gray')
#         ax1.set_title("FAF (Ch 0)")
#         ax1.axis('off') # Hide axes for cleaner image visualization

#         # Plot 2: Mask (Channel 1)
#         ax2 = axes[1]
#         ax2.imshow(mask, cmap='gray')
#         ax2.set_title("Mask (Ch 1)")
#         ax2.axis('off')

#         # Plot 3: Residual/Growth Mask (Channel 2)
#         ax3 = axes[2]
#         ax3.imshow(residual, cmap='gray')
#         ax3.set_title("Growth Mask (Ch 2)")
#         ax3.axis('off')

#         # Adjust layout to prevent titles from overlapping
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.show()

#     print(f"\nVisualization complete for sample index {sample_idx} from the {dataset_name} set across all four time steps.")


def visualize_sample(train_dataset, test_dataset, sample_idx, dataset_name='test'):
    """
    Visualizes the three channels (FAF, Mask, Residual) for all 4 time steps 
    of a single sample. Time flows from left to right.
    """
    
    if dataset_name == 'train':
        dataset = train_dataset
    elif dataset_name == 'test':
        dataset = test_dataset
    else:
        print(f"Error: Invalid dataset_name '{dataset_name}'.")
        return

    if len(dataset) <= sample_idx:
        print(f"{dataset_name.capitalize()} dataset too small for index {sample_idx}.")
        return

    # Data_item shape: [3, 4, 256, 256] (Channels, Time, H, W)
    Data_item = dataset[sample_idx]

    # Create a grid: 3 Rows (Channels), 4 Columns (Time Steps)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(f'Sample Index {sample_idx} ({dataset_name.upper()} SET) - Temporal Overview', fontsize=20, y=1.02)

    channel_labels = [
        "FAF Image", 
        "Lesion Mask", 
        "Growth Mask"
    ]

    for row_idx in range(3):       # Iterate through Channels
        for col_idx in range(4):   # Iterate through Time (Frames)
            # Extract data [Channel, Time, H, W]
            img = Data_item[row_idx, col_idx, :, :].cpu().detach().numpy()
            
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap='gray')
            
            # Label: "Channel Name - Frame i" (where i is 1 to 4)
            ax.set_title(f"{channel_labels[row_idx]} - Month {col_idx*6}", fontsize=15)
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def compare_split_masks(train_dataset, test_dataset, channel_type='mask', time_step=0):
    """
    Displays grids of the Mask or Residual channel for a specified time step (T=0 to 3), 
    showing only the unaugmented (original) samples from the training and testing sets.

    Args:
        train_dataset (Dataset): The training dataset.
        test_dataset (Dataset): The testing dataset.
        channel_type (str): 'mask' (Channel 1) or 'residual' (Channel 2).
        time_step (int): The time index of the frame to visualize (0, 1, 2, or 3).
    """
    
    # 1. Input Validation: Determine channel index and name
    if channel_type == 'residual':
        channel_idx = 2
        channel_name = "Growth"
    elif channel_type == 'mask':
        channel_idx = 1
        channel_name = "Mask"
    else:
        print(f"Error: Invalid channel_type '{channel_type}'. Must be 'mask' or 'residual'.")
        return
    
    if time_step not in range(4):
        print(f"Error: Invalid time_step '{time_step}'. Must be between 0 and 3.")
        return
    
    # Assumed sampling rate to select only original (unaugmented) samples
    SAMPLES_PER_GROUP = 10 

    # --- Helper function to plot a collection of masks in a grid ---
    def plot_masks_grid(dataset, title, indices_to_plot, cols, rows, current_channel_idx, current_time_step):
        num_plots = len(indices_to_plot)
        
        fig_width = cols * 3.5
        fig_height = rows * 3.5

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
        fig.suptitle(title, fontsize=24)
        
        for plot_num, i in enumerate(indices_to_plot):
            row = plot_num // cols
            col = plot_num % cols
            
            # Extract the data tensor slice [1, H, W] and convert to numpy
            data_t = dataset[i][current_channel_idx, current_time_step, :, :].cpu().detach().numpy()
            
            ax = axes[row, col]
            ax.imshow(data_t, cmap='gray')
            ax.set_title(f'Sample {i}', fontsize=8) 
            ax.axis('off')

        # Hide any subplots that are not used by the data
        for i in range(num_plots, rows * cols):
            axes.flatten()[i].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()

    # --- Plot Training Data ---
    N_TRAIN = len(train_dataset)
    # Select indices that correspond to the start of a group (unaugmented samples)
    train_original_indices = np.arange(0, N_TRAIN, SAMPLES_PER_GROUP)
    N_TRAIN_ORIGINAL = len(train_original_indices)
    
#     plot_masks_grid(train_dataset, 
#                     f"Training Set {channel_name} (T={time_step}) - Original Samples: {N_TRAIN_ORIGINAL} / {N_TRAIN} total", 
#                     train_original_indices,
#                     cols=13, 
#                     rows=4,
#                     current_channel_idx=channel_idx,
#                     current_time_step=time_step)
    plot_masks_grid(train_dataset, 
                f"Training Set {channel_name} (T={time_step}) - Original Samples: {N_TRAIN_ORIGINAL} / {N_TRAIN} total", 
                train_original_indices,
                cols=13, 
                rows=5, # <--- INCREASED ROWS to 5 (accommodates up to 65 samples)
                current_channel_idx=channel_idx,
                current_time_step=time_step)

    # --- Plot Testing Data ---
    N_TEST = len(test_dataset)
    # Select indices that correspond to the start of a group (unaugmented samples)
    test_original_indices = np.arange(0, N_TEST, SAMPLES_PER_GROUP)
    N_TEST_ORIGINAL = len(test_original_indices)
    
    plot_masks_grid(test_dataset, 
                    f"Testing Set {channel_name} (T={time_step}) - Original Samples: {N_TEST_ORIGINAL} / {N_TEST} total", 
                    test_original_indices,
                    cols=7, 
                    rows=2,
                    current_channel_idx=channel_idx,
                    current_time_step=time_step)

    print(f"\nVisual comparison complete. Two grids (Train and Test) showing only the original (unaugmented) {channel_name.lower()}s at Time Step T={time_step} have been generated.")