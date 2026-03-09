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
    def __init__(self, in_channels, mid_channels: int=128, out_channels: int=1):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.base_patch = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels),
            nn.Conv2d(mid_channels, mid_channels // 4, 1),
            nn.ReLU(inplace=True)
        )

        self.receptive_1 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=1, dilation=1)
        self.receptive_2 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=2, dilation=2)
        self.receptive_3 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=3, dilation=3)
        self.receptive_4 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=4, dilation=4)

        self.end = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, 1),
            nn.Softplus(),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.head(x)

        x1 = self.receptive_1(x)
        x2 = self.receptive_2(x)
        x3 = self.receptive_3(x)
        x4 = self.receptive_4(x)

        F_mid = torch.cat([x1, x2, x3, x4], dim=1)
        density_out = self.end(F_mid)

        return density_out


# ======= 20260309 ======
class DensityPredictorP(nn.Module):
    def __init__(self, in_channels, mid_channels=128, out_channels=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.base = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels),
            nn.Conv2d(mid_channels, mid_channels // 4, 1),
            nn.SiLU(inplace=True)
        )

        self.dila1 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=1, dilation=1)
        self.dila2 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=2, dilation=2)
        self.dila3 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=3, dilation=3)
        self.dila4 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=4, dilation=4)

        self.end = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.head(x)

        x_base = self.base(x)
        x1 = self.dila1(x)
        x2 = self.dila2(x) + x_base
        x3 = self.dila3(x) + x_base
        x4 = self.dila4(x) + x_base

        mid = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.end(mid)

        return out


class PredictorPBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.block1 = DensityPredictorP(in_channels=in_channels, out_channels=mid_channels)
        self.block2 = DensityPredictorP(in_channels=in_channels, out_channels=out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1),
            nn.Softplus()
        )
        self.conv2 = nn.Conv2d(mid_channels + in_channels, in_channels, 1)

    def forward(self, x):
        x1_out = self.block1(x)
        x1_out_1 = self.conv1(x1_out)

        mid = self.conv2(torch.cat([x, x1_out], dim=1))
        x2_out = self.block2(mid)

        return x1_out_1, x2_out


class ConfidencePredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        super().__init__()
        self.head = nn.Conv2d(in_channels, 64, 1)
        self.body = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        x1 = self.head(x)
        out = self.body(x1)

        return out