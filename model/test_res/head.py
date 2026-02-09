import torch
import torch.nn as nn


class HeatmapHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),

            # nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
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
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            # nn.ReLU(inplace=True),
        )

        self.receptive_1 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=1, dilation=1)
        self.receptive_2 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=2, dilation=2)
        self.receptive_3 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=3, dilation=3)
        self.receptive_4 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=4, dilation=4)

        self.end = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 1),
            nn.ReLU(inplace=True),
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


# class ResPredictor(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(in_channels // 2, out_channels, 1)
#         )
#
#     def forward(self, Ci_mid):
#         res_out = self.conv(Ci_mid)
#
#         return res_out

class ResPredictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

        nn.init.constant_(self.conv[-1].weight, 0)
        nn.init.constant_(self.conv[-1].bias, 0)

    def forward(self, Ci_mid):
        res_out = self.conv(Ci_mid)

        return res_out