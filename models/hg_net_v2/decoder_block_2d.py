import torch
from torch import nn

from models.hg_net_v2.conv_bn_act_2d import ConvBnAct2d

class DecoderBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_connection_channels: int,
        out_channels: int,
        scale_factor: int = 2,
    ):
        super().__init__()

        self._upsample = nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=in_channels,
                                            kernel_size=scale_factor,
                                            stride=scale_factor,
        )

        k = 3
        c = skip_connection_channels if skip_connection_channels != 0 else in_channels
        self._intermediate_conv = nn.Sequential(ConvBnAct2d(c, c, k, k//2),
                                                ConvBnAct2d(c, c, k, k//2))

        self._conv1 = ConvBnAct2d(in_channels + skip_connection_channels,
                                  out_channels,
                                  kernel_size= k,
                                  padding= k//2)

        self._conv2 = ConvBnAct2d(out_channels,
                                  out_channels,
                                  kernel_size= k,
                                  padding= k//2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor=None) -> torch.Tensor:
        x = self._upsample(x)

        if skip is not None:
            skip = self._intermediate_conv(skip)
            x = torch.cat([x, skip], dim=1)
        else:
            x = self._intermediate_conv(x)

        x = self._conv1(x)
        x = self._conv2(x)
        return x
