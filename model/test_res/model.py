from .conv import *
from .head import *
from .module import *


class UNetTestRes(nn.Module):
    def __init__(self, in_channels, num_class=2, bilinear=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_class
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (DownScaling(64, 128))
        self.down2 = (DownScaling(128, 256))
        self.down3 = (DownScaling(256, 512))

        factor = 2 if bilinear else 1
        self.down4 = (DownScaling(512, 1024 // factor))
        self.up1 = (UpScaling(1024, 512 // factor, bilinear))
        self.up2 = (UpScaling(512, 256 // factor, bilinear))
        self.up3 = (UpScaling(256, 128 // factor, bilinear))
        self.up4 = (UpScaling(128, 64, bilinear))

        self.relu = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1)
        self.density_conv = nn.Conv2d(3, 64, 1)

        self.heatmap_head = HeatmapHead(in_channels=64, out_channels=num_class)

        self.density_predictor = DensityPredictor(in_channels=64, out_channels=1)
        self.res_predictor = ResPredictor(in_channels=3, out_channels=2)

        self.sigmoid = nn.Sigmoid()

        self.k = nn.Parameter(torch.zeros(1))

    def forward(self, Ci):
        # Down
        x1 = self.inc(Ci)   # 64
        x2 = self.down1(x1) # 128  /2
        x3 = self.down2(x2) # 256  /4  @@
        x4 = self.down3(x3) # 512  /8
        x5 = self.down4(x4) # 512  /16 @@

        # Up
        Fi = self.up1(x5, x4)
        Fi = self.up2(Fi, x3)
        Fi = self.up3(Fi, x2)
        Fi = self.up4(Fi, x1)

        ## ======
        # density_out = self.density_predictor(Fi)  # 1
        #
        # Ci_avg = subtract_avg_pool(Ci, k=3)  # 3
        # Ci_mid = Ci_avg * self.sigmoid(Ci_avg) + Ci_avg
        # gate = Ci_mid * torch.sigmoid(density_out) # 3 * 1 = 3
        # Fi_res = self.res_predictor(gate)  # 2
        #
        # density_out = density_out * Ci_mid # 1 * 3 = 3
        # density_feature = self.density_conv(density_out)  # @@.
        # Fi_density = Fi * density_feature
        # heatmap_out = self.heatmap_head(Fi + Fi_density)  # main head
        # heatmap_out = heatmap_out + self.k * Fi_res

        ## ======20260209
        # density_out = self.density_predictor(Fi) # 1
        # gate = torch.sigmoid(density_out)
        #
        # Ci_avg = subtract_avg_pool(Ci, k=3) # 3
        # Ci_mid = Ci_avg * self.sigmoid(Ci_avg) + Ci_avg
        # Ci_mid_1 = torch.mean(Ci_mid, dim=1, keepdim=True)
        #
        # density_f = Ci_mid_1 * gate # 1
        #
        # Fi_density = Fi * density_f
        # heatmap_out = self.heatmap_head(Fi + Fi_density) # main head 2
        #
        # res_input = torch.cat([heatmap_out, density_f], dim=1)
        # Fi_res = self.res_predictor(res_input) # 2
        # heatmap_out = heatmap_out + self.k * Fi_res

        ## ======
        density_out = self.density_predictor(Fi)  # 1
        Ci_avg = subtract_avg_pool(Ci, k=3)  # 3
        # Ci_mid = Ci_avg * self.sigmoid(Ci_avg) + Ci_avg

        gate = self.density_conv(density_out * Ci_avg)
        Fi_enhanced = Fi * (1 + gate)
        heatmap_out = self.heatmap_head(Fi_enhanced)

        return {
            "heatmap_out": heatmap_out, # [B, 2, H, W]
        }

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)