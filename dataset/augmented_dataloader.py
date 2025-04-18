import os
import glob
import cv2
import torch
from torch.utils.data import Dataset

class OnDiskSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, to_tensor=True, exts=('jpg','png')):
        """
        image_dir:  path to folder containing RGB images
        mask_dir:   path to folder containing corresponding segmentation masks
        to_tensor:  if True, convert numpy arrays to torch.Tensor
        exts:       tuple of allowed image file extensions (without dot)
        """
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.to_tensor = to_tensor
        # collect image filenames with allowed extensions
        patterns = [os.path.join(image_dir, f"*.{ext}") for ext in exts]
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        # store basenames (e.g. 'polyp_3.jpg') sorted
        self.images = sorted([os.path.basename(f) for f in files])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fn = self.images[idx]
        img_path = os.path.join(self.image_dir, fn)
        mask_path = os.path.join(self.mask_dir, fn)

        # Load image (BGR) and convert to RGB
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure binary mask (0.0 or 1.0)
        mask = (mask > 127).astype('float32')

        if self.to_tensor:
            # HWC → CHW, scale image to [0,1]
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)
            # add channel dimension to mask (1×H×W)
            mask = torch.from_numpy(mask).unsqueeze(0)

        return {
            'sat_img': img,
            'map_img': mask,
            'filename': fn
        }


def get_on_disk_datasets(root_dir, to_tensor=True, exts=('jpg','png')):
    """
    root_dir structure:
      train/images, train/masks
      valid/images, valid/masks
      test/images,  test/masks

    Returns: (train_ds, valid_ds, test_ds)
    """
    def build(split):
        img_dir  = os.path.join(root_dir, split, 'images')
        mask_dir = os.path.join(root_dir, split, 'masks')
        return OnDiskSegDataset(img_dir, mask_dir, to_tensor=to_tensor, exts=exts)

    train_ds = build('train')
    valid_ds = build('valid')
    test_ds  = build('test')
    return train_ds, valid_ds, test_ds
