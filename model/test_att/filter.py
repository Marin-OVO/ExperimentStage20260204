import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletFrequencyExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        kernel = torch.tensor([
            [[[1, 1], [1, 1]]],
            [[[1, -1], [1, -1]]],
            [[[1, 1], [-1, -1]]],
            [[[1, -1], [-1, 1]]]
        ], dtype=torch.float32) / 2.0

        self.register_buffer('weight', kernel.repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        x_wavelet = F.conv2d(x, self.weight, stride=2, groups=x.shape[1])

        B, C4, H, W = x_wavelet.shape
        x_reshaped = x_wavelet.view(B, -1, 4, H, W)
        high_freq = x_reshaped[:, :, 1:, :, :]

        out = torch.mean(high_freq, dim=2)

        return out