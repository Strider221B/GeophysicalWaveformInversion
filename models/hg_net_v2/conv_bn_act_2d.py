from typing import Tuple, Union

import torch
from torch import nn

class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        padding: Union[int, Tuple] = 0,
        stride: Union[int, Tuple] = 1,
    ):
        super().__init__()

        self._conv= nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False,
        )
        self._activation= nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        x = self._activation(x)
        return x
