import torch
from pathlib import Path
from datetime import datetime
import os

## Save the model

def save_model_weights(
    model, 
    final_epoch, 
    save_dir, 
    model_name
):
    """
    Saves the model state and metadata (epoch) to a specified directory.

    Args:
        model: The PyTorch model instance (e.g., model_baseline).
        final_epoch: The epoch number to record in the checkpoint.
        save_dir: The full Path object pointing to the target directory.
        model_name: A base string name for the saved file.

    Returns:
        The save_path if successful, otherwise None.
    """
    try:
        # 1. Ensure the directory exists
        save_dir.mkdir(parents=True, exist_ok=True)

        # 2. Create a unique filename with timestamp and final epoch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_epoch{final_epoch}_{timestamp}.pth"
        save_path = save_dir / filename

        # 3. Create the state dictionary 
        checkpoint = {
            'epoch': final_epoch,
            'model_state_dict': model.state_dict(),
            'description': 'Final weights after synthetic pretraining phase.'
        }

        # 4. Save the checkpoint
        torch.save(checkpoint, save_path)
        print(f"\n Model successfully saved to: {save_path.absolute().as_posix()}")
        return save_path
        
    except Exception as e:
        print(f"\n Error saving model: {e}")
        return None
 

def load_model(model, model_path, device):
    """
    Loads model weights from a checkpoint file, maps to the correct device, 
    and sets the model to evaluation mode.

    Args:
        model: The instantiated model architecture (e.g., model_baseline).
        model_path: The full Path object to the .pth checkpoint file.
        device: The target device (e.g., args.device).

    Returns:
        A tuple containing (loaded_model, loaded_epoch_number).
    """
    loaded_epoch = 'N/A'
    
    if not os.path.exists(model_path):
        print(f"\n Error: Checkpoint file not found at {model_path}")
        return model, loaded_epoch
    
    try:
        # Load the checkpoint dictionary from the path, mapping it to the correct device
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load the state dictionary into the instantiated model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Optionally retrieve the epoch number
        loaded_epoch = checkpoint.get('epoch', 'N/A')
        
        # Set the model to evaluation mode (important after loading for fine-tuning)
        model.eval()
        
        print(f"\n Model weights successfully loaded from: {model_path.name}")
        print(f"   Model restored to state after epoch: {loaded_epoch}")
        print("   Model is set to evaluation mode (model.eval()).")
        
        return model, loaded_epoch
        
    except Exception as e:
        print(f"\n Failed to load model weights: {e}")
        return model, loaded_epoch