import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from utils.hparams import HParam
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus

from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import Dataset, DataLoader


class KvasirSegDatasetPredict(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_dir = image_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)
        image_tensor = transformed["image"]

        return {
            "image": image_tensor,
            "original": image,
            "filename": img_name
        }


def predict(hp, checkpoint_path, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = ResUnetPlusPlus(3).cuda() if hp.RESNET_PLUS_PLUS else ResUnet(3, 64).cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Data
    dataset = KvasirSegDatasetPredict(input_dir, image_size=hp.IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Running prediction on {len(dataset)} images...")

    with torch.no_grad():
        for batch in tqdm(loader):
            image = batch["image"].cuda()
            original = batch["original"][0].numpy()
            original_resized = cv2.resize(original, (hp.IMAGE_SIZE, hp.IMAGE_SIZE))

            filename = batch["filename"][0]
            basename = os.path.splitext(filename)[0]

            # Forward pass
            output = model(image)
            print("Model output shape:", output.shape)
            pred_mask = (torch.sigmoid(output) > 0.5).float().cpu().numpy()[0, 0] * 255
            pred_mask = pred_mask.astype(np.uint8)

            # Save original input (resized to 256x256)
            input_path = os.path.join(output_dir, f"{basename}_input.png")
            cv2.imwrite(input_path, cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR))
            
            # Save predicted mask (already 256x256)
            mask_path = os.path.join(output_dir, f"{basename}_pred_mask.png")
            cv2.imwrite(mask_path, pred_mask)
            
            # Save side-by-side view (both 256x256)
            combined = np.concatenate([
                original_resized,
                cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
            ], axis=1)
            combined_path = os.path.join(output_dir, f"{basename}_combined.png")
            cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polyp Segmentation Inference")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input image folder")
    parser.add_argument("--output", type=str, required=True, help="Folder to save predictions")

    args = parser.parse_args()
    hp = HParam(args.config)

    predict(hp, args.checkpoint, args.input, args.output)
