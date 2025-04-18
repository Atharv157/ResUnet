import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
from your_dataset_module import get_on_disk_datasets  # Import your dataset class

def save_predictions(model, test_loader, device, output_dir="predictions"):
    """Generate and save predictions for test dataset"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['sat_img'].to(device)
            filenames = batch['filename']
            
            # Forward pass
            outputs = model(images)
            
            # Convert outputs to probabilities and binary masks
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            
            # Save predictions
            for i in range(preds.shape[0]):
                mask = preds[i].squeeze().cpu().numpy()  # Remove batch and channel dim
                mask = (mask * 255).astype(np.uint8)     # Convert to 0-255 range
                
                # Save with original filename
                cv2.imwrite(
                    os.path.join(output_dir, filenames[i]),
                    mask
                )

if __name__ == "__main__":
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = "/content/drive/MyDrive/polyp_segmentation/checkpoints/ThirdExpt/ThirdExpt_checkpoint_epoch20.pt"
    DATA_ROOT = "new_data/polyp-dataset/"
    BATCH_SIZE = 4
    
    # Load model
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    model = model.to(DEVICE)
    
    # Get datasets
    _, _, test_ds = get_on_disk_datasets(
        root_dir=DATA_ROOT,
        to_tensor=True,
        exts=('jpg', 'png')
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Generate and save predictions
    save_predictions(model, test_loader, DEVICE, output_dir="/content/drive/MyDrive/polyp_segmentation/checkpoints/ThirdExpt/predictions")
    
    print("Prediction completed. Results saved in 'predictions' directory")