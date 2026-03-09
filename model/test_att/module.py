import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]

    return p

def subtract_avg_pool(x, k=5):
    avg = F.avg_pool2d(
        x, kernel_size=k, stride=1, padding=k//2
    )

    return x - avg


class MaxPoolConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel: int=1, stride: int=1, padding=None, groups: int=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, autopad(kernel, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):

        return self.act(self.bn(self.conv(x)))


class MaxPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel: int=3, e=0.5):
        super().__init__()
        c_ = int(in_channels * e)

        self.conv1 = MaxPoolConv(in_channels, c_, 1, 1)
        self.conv2 = MaxPoolConv(in_channels, c_, 1, 1)
        self.conv3 = MaxPoolConv(c_, c_, 3, 1)
        self.conv4 = MaxPoolConv(c_, c_, 1, 1)

        self.maxpool = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)

        self.conv5 = MaxPoolConv(c_ * 4, c_, 1, 1)
        self.conv6 = MaxPoolConv(c_ * 2, out_channels, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x1)
        x3 = self.conv4(x2)

        pool1 = self.maxpool(x3)
        pool2 = self.maxpool(pool1)
        pool3 = self.maxpool(pool2)

        pool = self.conv5(torch.cat([x1, pool1, pool2, pool3], 1))
        y1 = self.conv3(pool)
        y2 = self.conv4(y1)

        return self.conv6(torch.cat([y2, self.conv2(x)], 1))


class RSAtt(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.map_avg = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )
        self.map_max = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_max = F.max_pool2d(x, kernel_size=3, stride=1, padding=1) - x

        gate = self.sigmoid(self.map_avg(x_avg) + self.map_max(x_max))

        return gate


class Gate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.gamma = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, x):
        gate = torch.sigmoid(self.conv(x))
        gate = torch.pow(gate, self.gamma)

        return gate


class SharpnessRefiner(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.laplacian_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                        padding=1, groups=in_channels, bias=False)

        kernel = torch.tensor([[-1., -1., -1.],
                               [-1., 8., -1.],
                               [-1., -1., -1.]])
        kernel = kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        self.laplacian_conv.weight.data = kernel
        self.laplacian_conv.weight.requires_grad = False

    def forward(self, x):
        sharp = self.laplacian_conv(x)
        gate = torch.sigmoid(sharp)

        return 1 + gate


class Weight(nn.Module):
    def __init__(self):
        super().__init__()


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pre_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.d1 = nn.Conv2d(out_channels, out_channels // 4, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(out_channels, out_channels // 4, 3, padding=2, dilation=2)
        self.d3 = nn.Conv2d(out_channels, out_channels // 4, 3, padding=3, dilation=3)
        self.d4 = nn.Conv2d(out_channels, out_channels // 4, 3, padding=4, dilation=4)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        self.post_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pre_conv(x)

        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        x4 = self.d4(x)
        x_ms = torch.cat([x1, x2, x3, x4], dim=1)

        k = self.se(x_ms)
        x_refined = x_ms * k

        out = self.post_conv(x_refined)
        out = self.bn(out)

        if x.shape == out.shape:
            out += x

        return self.relu(out)


class Branch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.block1 = Block(64, 64)
        self.block2 = Block(64, 64)

    def forward(self, x, mask):
        x1 = self.conv(x)
        mask = torch.sigmoid(mask)

        x1 = self.block1(x1) * mask
        out = self.block2(x1) * mask

        return out

