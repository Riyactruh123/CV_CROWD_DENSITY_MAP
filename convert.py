import os
import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import gaussian_filter

def generate_density_map(image_shape, points, sigma=15):
    density = np.zeros(image_shape, dtype=np.float32)
    if len(points) == 0:
        return density
    h, w = image_shape

    for pt in points:
        x = min(w - 1, max(0, int(pt[0])))
        y = min(h - 1, max(0, int(pt[1])))
        density[y, x] = 1

    return gaussian_filter(density, sigma=sigma)

# Paths
mat_dir = 'dataset/ShanghaiTech/part_A/train_data/ground-truth'
img_dir = 'dataset/ShanghaiTech/part_A/train_data/images'
output_dir = 'dataset/ShanghaiTech/part_A/train_data/density_map'

os.makedirs(output_dir, exist_ok=True)

for mat_file in tqdm(os.listdir(mat_dir), desc="Converting MAT to H5"):
    if not mat_file.endswith('.mat'):
        continue

    try:
        index = mat_file.replace('GT_IMG_', '').replace('.mat', '')
        img_file = f"processed_IMG_{index}.jpg"
        img_path = os.path.join(img_dir, img_file)

        if not os.path.exists(img_path):
            print(f"⚠️ Image not found for {img_file}")
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        shape = (h, w)

        mat = sio.loadmat(os.path.join(mat_dir, mat_file))

        if 'density' in mat:
            density = mat['density']
        elif 'image_info' in mat:
            points = mat['image_info'][0][0][0][0][0]  # (x, y) format
            density = generate_density_map(shape, points)
        else:
            print(f"❗ Unknown format in {mat_file}")
            continue

        h5_path = os.path.join(output_dir, f"processed_IMG_{index}.h5")
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('density', data=density)

    except Exception as e:
        print(f"❌ Failed to process {mat_file}: {e}")
