import os
import torch
from torch.utils.data import DataLoader
from datasets.crowd_datasets import CrowdDataset
from models.model import PCCNet  # Ensure this matches your model's class
import torch.nn.functional as F

def main():
    # Set dataset paths
    image_dir = "/Users/kushireddy/Downloads/shanghaitech_part_B/train_data/img"
    density_dir = "/Users/kushireddy/Downloads/shanghaitech_part_B/train_data/den"
    mask_dir = "/Users/kushireddy/Downloads/shanghaitech_part_B/train_data/seg"  # Optional

    # Create dataset and dataloader
    train_dataset = CrowdDataset(image_dir, density_dir, mask_dir=mask_dir, downsample_factor=4)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Setup model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PCCNet().to(device)
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(2):  # Adjust epoch count as needed
        for i, batch in enumerate(train_loader):
            if len(batch) == 2:
                images, targets = batch
                masks = None
            elif len(batch) == 3:
                images, targets, masks = batch
            else:
                raise ValueError("Unexpected number of items in batch.")

            images, targets = images.to(device), targets.to(device)
            if masks is not None:
                masks = masks.to(device)

            # Add channel dim if needed
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)

            # Forward pass
            raw_output = model(images)
            outputs = raw_output[0] if isinstance(raw_output, tuple) else raw_output

            # Debug shape
            print(f"Output shape: {outputs.shape}, Target shape: {targets.shape}")

            # Compute loss and update
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/2], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            torch.save(model.state_dict(), "pccnet_final.pth")
print("📁 Model saved as 'pccnet_final.pth'")

if __name__ == "__main__":
    main()