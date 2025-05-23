import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import CrowdDataset
import torchvision.transforms as transforms
import os

# Dummy CNN model (replace with your actual model if needed)
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

# Configurations
image_dir = '/Users/kushireddy/SINet/dataset/ShanghaiTech/part_A/train_data/images'
density_dir = '/Users/kushireddy/SINet/dataset/ShanghaiTech/part_A/train_data/density_map'
batch_size = 4
epochs = 5
lr = 1e-4
resize_to = (224, 224)
model_save_path = 'crowd_model.pth'

# Check if dataset path exists
if not os.path.exists(image_dir) or not os.path.exists(density_dir):
    raise FileNotFoundError("Dataset folders not found. Check 'dataset/train/images' and 'dataset/train/density_maps'.")

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = CrowdDataset(image_dir=image_dir, density_dir=density_dir, transform=transform, output_size=resize_to)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model, optimizer, loss
model = DummyCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, densities in dataloader:
        images = images.to(device)
        densities = densities.to(device)

        preds = model(images)
        loss = criterion(preds, densities)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ Model saved successfully to '{model_save_path}'")
