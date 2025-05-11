from typing import Tuple, Union

import torch
from torch import nn

class SegmentationHead2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: Union[int, Tuple[int]] = (2,2),
        kernel_size: int = 3,
    ):
        super().__init__()
        self._conv= nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self._upsample = nn.Upsample(scale_factor= scale_factor, mode='bilinear', align_corners= False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        x = self._upsample(x)
        return x
