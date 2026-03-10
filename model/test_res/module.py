import torch
import torch.nn as nn
import torch.nn.functional as F


def subtract_avg_pool(x, k=5):
    avg = F.avg_pool2d(
        x, kernel_size=k, stride=1, padding=k//2
    )

    return x - avg


# ============
class AdaptiveAvgPoolKernel(nn.Module):
    def __init__(self, in_channels, out_channels, k_list=[3, 5, 7]):
        super().__init__()
        self.k_list = k_list
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * len(k_list), out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        x_avg_list = []
        for k in self.k_list:
            x_avg = subtract_avg_pool(x, k)
            x_avg_list.append(x_avg)

        x_avg_cat = torch.cat(x_avg_list, dim=1)

        return self.conv(x_avg_cat)
# ============