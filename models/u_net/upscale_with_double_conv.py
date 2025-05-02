import torch
import torch.nn as nn
import torch.nn.functional as F

from models.u_net.double_conv import DoubleConv

class UpscaleWithDoubleConv(nn.Module):
    """Upscaling then DoubleConv"""

    def __init__(self, in_channels, out_channels, use_bilinear=True):
        super().__init__()
        self._use_bilinear = use_bilinear

        if use_bilinear:
            self._up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            )
            # In bilinear mode, input channels to conv is sum of skip and upsampled
            conv_in_channels = in_channels + out_channels
            # Mid channels in DoubleConv is explicitly set to out_channels
            self._conv = DoubleConv(
                conv_in_channels, out_channels, mid_channels=out_channels
            )
        else:
            # Use ConvTranspose2d for learned upsampling
            self._up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            # Input channels to conv is sum of skip and halved upsampled channels
            conv_in_channels = (in_channels // 2) + out_channels
            self._conv = DoubleConv(conv_in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is the feature map from the layer below (needs upsampling)
        # x2 is the skip connection from the corresponding encoder layer
        x1 = self._up(x1)

        # Pad x1 if its dimensions don't match x2 after upsampling
        # Input is CHW
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        # Pad format: (padding_left, padding_right, padding_top, padding_bottom)
        x1 = F.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self._conv(x)
