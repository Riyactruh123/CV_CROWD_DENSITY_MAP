import os
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize as tv_resize

class CrowdDataset(Dataset):
    def __init__(self, image_dir, density_dir, transform=None, output_size=(224, 224)):
        self.image_dir = image_dir
        self.density_dir = density_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform or transforms.ToTensor()
        self.output_size = output_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        density_path = os.path.join(self.density_dir, img_name.replace('.jpg', '.h5'))

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        image = tv_resize(image, self.output_size)

        with h5py.File(density_path, 'r') as f:
            if 'density' not in f:
                raise KeyError(f"'density' not found in {density_path}")
            density = np.array(f['density'], dtype=np.float32)

        density = torch.from_numpy(density).unsqueeze(0)
        density = tv_resize(density, self.output_size)

        return image, density
