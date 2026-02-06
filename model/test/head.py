import torch
import torch.nn as nn


class HeatmapHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        heatmap_out = self.conv(x)

        return heatmap_out


class DensityPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels=128, out_channels=1):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.receptive_1 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, dilation=1)
        self.receptive_2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=2, dilation=2)
        self.receptive_3 = nn.Conv2d(mid_channels, mid_channels, 3, padding=3, dilation=3)

        self.end = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, 1),
            nn.Softplus(),
            # nn.Sigmoid()
        )

    def forward(self, x, k):
        x = self.head(x)

        x1 = self.receptive_1(x)
        x2 = self.receptive_2(x)
        x3 = self.receptive_3(x)

        F_mid = (k[:, 0].view(-1, 1, 1, 1) * x1 +
                         k[:, 1].view(-1, 1, 1, 1) * x2 +
                         k[:, 2].view(-1, 1, 1, 1) * x3)

        density_out = self.end(F_mid)

        return density_out


class KWeights(nn.Module):
    def __init__(self, in_channels, out_channels: int=3, mid_channels: int=128):
        super().__init__()

        # self.head = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, 3, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        # )

        self.weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        k = self.weights(x)

        return k
