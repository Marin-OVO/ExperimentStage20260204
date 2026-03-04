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
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1)
        )
        self.map_max = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_max = F.max_pool2d(x, kernel_size=3, stride=1, padding=1) - x

        gate = self.sigmoid(self.map_avg(x_avg) + self.map_max(x_max))

        return gate
