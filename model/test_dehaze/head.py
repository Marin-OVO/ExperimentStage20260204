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
# class DensityPredictorP(nn.Module):
#     def __init__(self, in_channels, mid_channels=128, out_channels=1):
#         super().__init__()
#         self.head = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.SiLU(inplace=True),
#         )
#         self.base = nn.Sequential(
#             nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels),
#             nn.Conv2d(mid_channels, mid_channels // 4, 1),
#             nn.SiLU(inplace=True)
#         )
#
#         self.dila1 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=1, dilation=1)
#         self.dila2 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=2, dilation=2)
#         self.dila3 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=3, dilation=3)
#         self.dila4 = nn.Conv2d(mid_channels, mid_channels // 4, 3, padding=4, dilation=4)
#
#         self.end = nn.Sequential(
#             nn.Conv2d(mid_channels, mid_channels // 2, 1),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(mid_channels // 2, out_channels, 1),
#             nn.Softplus(),
#         )
#
#     def forward(self, x):
#         x = self.head(x)
#
#         x_base = self.base(x)
#         x1 = self.dila1(x)
#         x2 = self.dila2(x) + x_base
#         x3 = self.dila3(x) + x_base
#         x4 = self.dila4(x) + x_base
#
#         mid = torch.cat([x1, x2, x3, x4], dim=1)
#         out = self.end(mid)
#
#         return out


# class DensityPredictorP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#
#         mid = in_channels // 2
#         self.head = nn.Sequential(
#             nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
#             nn.BatchNorm2d(mid),
#             nn.ReLU(inplace=True),
#         )
#
#         self.base = nn.Sequential(
#             nn.Conv2d(mid, mid, 3, padding=1, groups=mid),
#             nn.Conv2d(mid, mid // 4, 1),
#             nn.SiLU(inplace=True)
#         )
#
#         self.dila1 = nn.Conv2d(mid, mid // 4, 3, padding=1, dilation=1)
#         self.dila2 = nn.Conv2d(mid, mid // 4, 3, padding=3, dilation=3)
#         self.dila3 = nn.Conv2d(mid, mid // 4, 3, padding=5, dilation=5)
#         self.dila4 = nn.Conv2d(mid, mid // 4, 3, padding=7, dilation=7)
#
#         self.end = nn.Sequential(
#             nn.Conv2d(mid, mid // 2, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid // 2, out_channels, 1),
#             nn.Softplus(),
#         )
#
#     def forward(self, x):
#         x = self.head(x)
#
#         x_base = self.base(x)
#         x1 = self.dila1(x) + x_base
#         x2 = self.dila2(x) + x_base
#         x3 = self.dila3(x) + x_base
#         x4 = self.dila4(x) + x_base
#
#         mid = torch.cat([x1, x2, x3, x4], dim=1)
#         out = self.end(mid)
#
#         return out


# class PredictorPBlock(nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels):
#         super().__init__()
#         self.block1 = DensityPredictorP(in_channels=in_channels, out_channels=mid_channels)
#         self.block2 = DensityPredictorP(in_channels=in_channels, out_channels=out_channels)
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(mid_channels, out_channels, 1),
#             nn.Softplus()
#         )
#         self.conv2 = nn.Conv2d(mid_channels + in_channels, in_channels, 1)
#
#     def forward(self, x):
#         x1_out = self.block1(x)
#         x1_out_1 = self.conv1(x1_out)
#
#         mid = self.conv2(torch.cat([x, x1_out], dim=1))
#         x2_out = self.block2(mid)
#
#         return x1_out_1, x2_out
# ======= 20260309 ======


class Maxout(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.out, -1, H, W)
        out, _ = torch.max(x, dim=1)

        return out


class Predictor(nn.Module):
    def __init__(self, in_channels, mid_channels=96, out_channels=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.base = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels),
            nn.Conv2d(mid_channels, mid_channels // 3, 1),
            nn.ReLU(inplace=True)
        )

        self.d1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 3, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels // 3),
            nn.ReLU(inplace=True)
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 3, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels // 3),
            nn.ReLU(inplace=True)
        )
        self.d3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 3, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels // 3),
            nn.ReLU(inplace=True)
        )

        self.end = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.head(x)

        x_base = self.base(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)

        mid = torch.cat([x1, x2, x3], dim=1)
        out = self.end(mid)

        return out


class DehazeRefine(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid = in_channels // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 5, padding=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 5, padding=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 7, padding=3, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 7, padding=3, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(3 * mid, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, 1),
            nn.Softplus(),
        )

        self.weight = WeightGenerator(mid)
    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(x)
        f3 = self.conv3(x)

        f1_weight, f2_weight, f3_weight = self.weight(f1, f2, f3)
        f1 = f1 * f1_weight
        f2 = f2 * f2_weight
        f3 = f3 * f3_weight

        f = torch.cat([f1, f2, f3], dim=1)
        out = self.out(f)

        return out


class WeightGenerator(nn.Module):
    def __init__(self, in_channels, out_channels: int=1):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3 * in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1)

        self.end1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.end2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.end3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, y, z):
        x_y_z_cat = torch.cat([x, y, z], dim=1)
        cat_out = self.base(x_y_z_cat)

        x_cat_out = self.conv1(cat_out)
        y_cat_out = self.conv2(cat_out)
        z_cat_out = self.conv3(cat_out)

        x_out = self.end1(x * x_cat_out)
        y_out = self.end2(y * y_cat_out)
        z_out = self.end2(z * z_cat_out)

        return x_out, y_out, z_out


class DehazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.predictor = Predictor(in_channels, out_channels=1)
        self.dehaze_refine = DehazeRefine(in_channels, out_channels)

        self.end = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 1),
            nn.Softplus(),
        )
        self.function = FunctionB()

    def forward(self, x):
        g = self.predictor(x)
        f = self.dehaze_refine(x)

        b = self.function(x, g)
        x1 = torch.clamp((x - b) / (f + 1e-6) + b, 0.1)
        out = self.end(x1)

        return f, out # >0


class FunctionB(nn.Module):
    def __init__(self):
        super().__init__()
        self.bg = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        self.k = nn.Parameter(torch.ones(1) * 0.5)
        self.proj = nn.Conv2d(64, 1, 1)

    def forward(self, x, mask):
        mu = torch.mean(x, dim=(2, 3), keepdim=True)
        sigma = torch.std(x, dim=(2, 3), keepdim=True)

        A = mu + self.k * sigma
        b = self.proj(A) * (1 - mask)

        return b


class DensityPredictorP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid = in_channels // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid, 5, padding=2, bias=False),
            nn.BatchNorm2d(mid),
            Maxout(4)
            # nn.ReLU(inplace=True),
        )
        self.base = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid),
            nn.Conv2d(mid, mid // 4, 1),
            nn.SiLU(inplace=True)
        )
        self.neck = nn.ModuleList([
            nn.Conv2d(mid, mid // 4, 3, padding=1, dilation=1),
            nn.Conv2d(mid, mid // 4, 3, padding=3, dilation=3),
            nn.Conv2d(mid, mid // 4, 3, padding=5, dilation=5),
            nn.Conv2d(mid, mid // 4, 3, padding=7, dilation=7)
        ])

        self.end = nn.Sequential(
            nn.Conv2d(mid, mid // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid // 2, out_channels, 1),
            nn.Sigmoid(),
        )
        self.end_maxout = nn.Sequential(
            nn.Conv2d(mid, mid // 2, 1),
            Maxout(4),
            nn.Conv2d(mid // 2, out_channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x_base = self.base(x)

        f = [conv(x) for conv in self.neck]

        mid = torch.cat(f, dim=1)
        out = self.end_maxout(mid)

        return out


class PredictorPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.head = DensityPredictorP(in_channels, 1)

        self.end = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 1),
            nn.Softplus(),
        )
        self.function = FunctionB()
    def forward(self, x):
        f = self.head(x) # Gamma

        b = self.function(x, f)
        x1 = (x - b) / (f + 1e-6) + b
        out = self.end(x1)

        return f, out

