import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.crowd_datasets import CrowdDataset
from models.model import PCCNet  # Update this import based on your model file name

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set paths
    image_dir = "/Users/kushireddy/Downloads/shanghaitech_part_B/test_data/img"
    density_dir = "/Users/kushireddy/Downloads/shanghaitech_part_B/test_data/den"
    mask_dir = "/Users/kushireddy/Downloads/shanghaitech_part_B/test_data/seg"  # Optional

    # Dataset and dataloader
    test_dataset = CrowdDataset(image_dir, density_dir, mask_dir=mask_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load model
    model = PCCNet()
    model.load_state_dict(torch.load("/Users/kushireddy/pcc_net2/pccnet_final.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # Metrics
    mae, mse = 0.0, 0.0

    os.makedirs("test_outputs", exist_ok=True)

    with torch.no_grad():
        for i, (images, targets, _) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            targets = targets.to(device)

            outputs, _ = model(images)

            # Resize targets if needed
            if outputs.shape[2:] != targets.shape[2:]:
                targets = F.interpolate(targets, size=outputs.shape[2:], mode='bilinear', align_corners=False)

            pred_count = outputs.sum().item()
            gt_count = targets.sum().item()

            mae += abs(pred_count - gt_count)
            mse += (pred_count - gt_count) ** 2

            print(f"[{i+1}/{len(test_loader)}] Predicted: {pred_count:.2f}, Ground Truth: {gt_count:.2f}")

            # Save predicted density map as image
            density_map = outputs.squeeze().cpu().numpy()
            plt.imshow(density_map, cmap='jet')
            plt.title(f"Pred: {pred_count:.1f} | GT: {gt_count:.1f}")
            plt.axis('off')
            plt.colorbar()
            plt.savefig(f"test_outputs/pred_{i+1:03d}.png", bbox_inches='tight')
            plt.close()

    mae /= len(test_loader)
    rmse = (mse / len(test_loader)) ** 0.5

    print(f"\n📊 Test MAE: {mae:.2f}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    test()
