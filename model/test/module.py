import torch
import torch.nn as nn
import torch.nn.functional as F


def subtract_avg_pool(x, k=5):
    avg = F.avg_pool2d(
        x, kernel_size=k, stride=1, padding=k//2
    )

    return x - avg


class AdaptiveKernel(nn.Module):
    def __init__(self, k_min: int = 3, k_max: int = 9, use_gmp: bool = True):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.use_gmp = use_gmp

    @torch.no_grad()
    def forward(self, heatmap: torch.Tensor):
        B = heatmap.size(0)

        gap = heatmap.mean(dim=(2, 3))
        if self.use_gmp:
            gmp = heatmap.amax(dim=(2, 3))
            stat = 0.5 * gap + 0.5 * gmp
        else:
            stat = gap

        score = stat[:, 0]
        score = torch.clamp(score, 0.0, 1.0)

        k_float = self.k_max - score * (self.k_max - self.k_min)
        k_int = k_float.round().int()

        k_int = torch.where(k_int % 2 == 0, k_int + 1, k_int)
        k_int = torch.clamp(k_int, self.k_min, self.k_max)

        return k_int.cpu().tolist() # 返回 List[int]
