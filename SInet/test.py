import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import resize as tv_resize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Dummy model — replace with your real model if needed
class DummyCNN(nn.Module):
    def __init__(self):
        super(DummyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.features(x)

# Paths
test_folder = '/Users/kushireddy/SINet/dataset/ShanghaiTech/part_A/test_data/images'
model_path = '/Users/kushireddy/SINet/crowd_model.pth'
output_folder = 'results'
resize_to = (224, 224)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DummyCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Get test images
image_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
if not image_files:
    print("❌ No .jpg files found in test_images/")
    exit()

print(f"🚀 Processing {len(image_files)} images (plotting + saving results)...\n")

# Process each image
for file_name in tqdm(image_files, desc="🔍 Testing"):
    img_path = os.path.join(test_folder, file_name)

    # Load image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img)
    img_tensor = tv_resize(img_tensor, resize_to)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        density_map = output.squeeze().cpu().numpy()
        count = density_map.sum()

    # Print to terminal
    print(f"🧍 {file_name}: Estimated Crowd Count = {count:.2f}")

    # Visualization
    plt.figure(figsize=(8, 3))
    plt.suptitle(f"{file_name} | Count: {count:.2f}", fontsize=12)

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Image")

    plt.subplot(1, 2, 2)
    plt.imshow(density_map, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.title("Density Map")

    plt.tight_layout()

    # Save plot
    save_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_result.png")
    plt.savefig(save_path)

    # Show and close
    plt.show(block=False)
    plt.pause(2)  # Show for 2 seconds
    plt.close()
