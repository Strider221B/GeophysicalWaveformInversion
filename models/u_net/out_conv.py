import torch.nn as nn

class OutConv(nn.Module):
    """1x1 Convolution for the output layer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self._conv(x)