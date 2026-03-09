from .conv import *
from .head import *
from .module import *
from .filter import *


class UNetTestAtt(nn.Module):
    def __init__(self, in_channels, num_class=2, bilinear=False):
        super(UNetTestAtt, self).__init__()
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

        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1)
        self.density_conv = nn.Conv2d(1, 64, 1)

        self.heatmap_head = HeatmapHead(in_channels=64, out_channels=num_class)
        self.density_predictor = DensityPredictorP(in_channels=64, out_channels=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # ====== Experiment ======
        self.gamma_filter = Gate(64)
        self.laplacian_filter = SharpnessRefiner(64)

        self.branch = Branch(3, 64)
        self.branch_weight = nn.Parameter(torch.ones(1) * 0.1)

        self.wfe = WaveletFrequencyExtractor(in_channels=3)
        self.wfe_conv = nn.Conv2d(3, 64, kernel_size=1)
        self.hf_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, 1),
            nn.Sigmoid()
        )
        self.hf_weight = nn.Parameter(torch.zeros(1))

        self.confidence_predictor = ConfidencePredictor(in_channels=64)
        self.confidence_conv = nn.Conv2d(1, 64, 1)

        self.predictor_block = PredictorPBlock(64, mid_channels=32, out_channels=1)
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

        # ====== Dila Branch ======
        # density_out = self.density_predictor(Fi) # 1
        # branch_Fi = self.branch(Ci, density_out)
        # ====== Dila Branch ======

        # ====== HF ======
        # hf = self.wfe(Ci)
        # hf = F.interpolate(hf, size=Fi.shape[2:], mode='bilinear', align_corners=True)
        # hf = self.wfe_conv(hf)
        # hf = hf * self.hf_se(hf)
        # Fi_refined = Fi + self.hf_weight * hf
        # density_out = self.density_predictor(Fi_refined)
        # ====== HF ======

        # density_out = self.density_predictor(Fi)  # 1

        # ====== confidence predictor ======
        # confidence_out = torch.sigmoid(self.confidence_predictor(Fi))
        # confidence_c64 = self.confidence_conv(confidence_out)
        # Fi_confidence = Fi * confidence_c64
        # density_out = self.density_predictor(Fi_confidence)  # 1
        # ====== confidence predictor ======

        # ====== predictor block ======
        x1_out, x2_out = self.predictor_block(Fi)
        x2_out_64 = self.density_conv(x2_out)
        heatmap_out = self.heatmap_head(Fi + Fi * x2_out_64)
        # ====== predictor block ======

        # ====== Max Pool =======
        # Fi_maxpool = self.maxpool_conv(Fi)
        # density_out = self.density_predictor(Fi_maxpool) # 1
        # ====== Max Pool =======

        # ====== adaptive kernel ======
        # k = self.kweights(Fi)
        # density_out = self.density_predictor(Fi, k) # 1
        # ====== adaptive kernel ======

        # ====== x - avg.pool ======
        # Ci_avg = subtract_avg_pool(Ci, k=3) # 3
        # Ci_mid = Ci_avg * self.sigmoid(Ci_avg) + Ci_avg
        # density_out = density_out * Ci_mid
        # ====== x - avg.pool ======

        # ====== Att gamma/laplacian ======
        # mask_gamma = self.gamma_filter(Fi_density)
        # mask_laplacian = self.laplacian_filter(Fi_density)
        # Fi_density = Fi_density * (mask_gamma * mask_laplacian)
        # ====== Att gamma/laplacian ======

        # density_feature = self.density_conv(density_out)
        # Fi_density = Fi * density_feature
        # heatmap_out = self.heatmap_head(Fi + Fi_density)

        return {
            "heatmap_out": heatmap_out,
            "x1_out": x1_out,
            "x2_out": x2_out,
            # "confidence_out": confidence_out,
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