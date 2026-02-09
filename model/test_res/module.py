import torch
import torch.nn as nn
import torch.nn.functional as F


def subtract_avg_pool(x, k=5):
    avg = F.avg_pool2d(
        x, kernel_size=k, stride=1, padding=k//2
    )

    return x - avg
