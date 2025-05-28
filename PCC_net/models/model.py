import torch.nn as nn
import torch.nn.functional as F

class PCCNet(nn.Module):
    def __init__(self):
        super(PCCNet, self).__init__()
        self.dme = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # downsample

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        self.output_density = nn.Conv2d(256, 1, 1)
        self.fbs = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.dme(x)
        density = self.output_density(features)
        fg_mask = self.fbs(features)
        return density, fg_mask
