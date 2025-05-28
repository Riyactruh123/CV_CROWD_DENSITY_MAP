import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

class CrowdDataset(Dataset):
    def __init__(self, image_dir, density_dir, mask_dir=None, downsample_factor=1):
        self.image_dir = image_dir
        self.density_dir = density_dir
        self.mask_dir = mask_dir
        self.downsample_factor = downsample_factor

        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.image_names.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        den_path = os.path.join(self.density_dir, img_name.replace('.jpg', '.csv'))

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        density = pd.read_csv(den_path, header=None).values.astype(np.float32)
        density = torch.from_numpy(density).unsqueeze(0)  # shape: [1, H, W]

        if self.downsample_factor > 1:
            H, W = density.shape[1:]
            H_ds, W_ds = H // self.downsample_factor, W // self.downsample_factor
            density = F.interpolate(density.unsqueeze(0), size=(H_ds, W_ds), mode='bilinear', align_corners=False).squeeze(0)

        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))
            mask = Image.open(mask_path).convert("L")
            mask = self.transform(mask)
            return image, density, mask

        return image, density
