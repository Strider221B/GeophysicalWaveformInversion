import torch.nn as nn

from models.u_net.double_conv import DoubleConv

class DownScaleWithDoubleConv(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self._maxpool_conv(x)