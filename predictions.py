import os
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from dataset.kvasir_dataloader import get_test_dataset
from utils.hparams import HParam

def save_mask(tensor, path):
    # Assumes tensor is [1, H, W], values in [0, 1]
    array = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(array).save(path)

def predict(model_path, image_dir, save_dir, config_path):
    os.makedirs(save_dir, exist_ok=True)

    hp = HParam(config_path)

    # Load model
    model = ResUnetPlusPlus(3).cuda() if hp.RESNET_PLUS_PLUS else ResUnet(3, 64).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Dataset
    test_dataset = get_test_dataset(image_dir=image_dir, image_size=hp.IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Inference
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            inputs = batch["sat_img"].cuda()
            filename = os.path.splitext(batch["filename"][0])[0] + ".png"

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            binary_mask = (preds > 0.5).float()

            save_path = os.path.join(save_dir, filename)
            save_mask(binary_mask[0], save_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Directory of input images")
    parser.add_argument("--output", required=True, help="Where to save predicted masks")
    args = parser.parse_args()

    predict(args.checkpoint, args.input, args.output, args.config)
