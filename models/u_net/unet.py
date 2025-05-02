import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.u_net.double_conv import DoubleConv
from models.u_net.downscale_with_double_conv import DownScaleWithDoubleConv
from models.u_net.out_conv import OutConv
from models.u_net.upscale_with_double_conv import UpscaleWithDoubleConv

class UNet(nn.Module):
    """U-Net architecture implementation"""

    def __init__(
        self,
        n_channels=Config.unet_in_channels,
        n_classes=Config.unet_out_channels,
        init_features=Config.unet_init_features,
        depth=Config.unet_depth,
        bilinear=Config.unet_bilinear,
    ):
        super().__init__()
        self._n_channels = n_channels
        self._n_classes = n_classes
        self._bilinear = bilinear
        self._depth = depth

        # Initial average pooling layer specific to this problem
        self._initial_pool = nn.AvgPool2d(kernel_size=(14, 1), stride=(14, 1))

        # --- Encoder ---
        self._encoder_blocks = nn.ModuleList()
        self._inc = DoubleConv(n_channels, init_features)
        self._encoder_blocks.append(self._inc)
        current_features = init_features
        for _ in range(depth):
            down_block = DownScaleWithDoubleConv(current_features, current_features * 2)
            self._encoder_blocks.append(down_block)
            current_features *= 2

        # --- Bottleneck ---
        # The last block of the encoder acts as the bottleneck
        bottleneck_features = current_features

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        current_features = bottleneck_features
        for _ in range(depth):
            # Output channels are halved at each step
            up_block = UpscaleWithDoubleConv(current_features, current_features // 2, bilinear)
            self.decoder_blocks.append(up_block)
            current_features //= 2

        # --- Output Layer ---
        self.outc = OutConv(current_features, n_classes)

    def forward(self, x):
        # Initial pooling and resizing
        x_pooled = self._initial_pool(x)
        x_resized = self._pad_or_crop(x_pooled, target_h=70, target_w=70)

        # --- Encoder Path ---
        skip_connections = []
        xi = x_resized
        for i, block in enumerate(self._encoder_blocks):
            xi = block(xi)
            # Store intermediate feature maps for skip connections (except bottleneck)
            if i < len(self._encoder_blocks) - 1:
                skip_connections.append(xi)

        # --- Decoder Path ---
        # Start with the bottleneck output (last xi)
        xu = xi
        # Iterate through decoder blocks and corresponding skip connections in reverse
        for i, block in enumerate(self.decoder_blocks):
            skip = skip_connections[len(skip_connections) - 1 - i]
            xu = block(xu, skip)

        # --- Final Output ---
        logits = self.outc(xu)
        # Apply scaling and offset specific to the problem's target range
        output = logits * 1000.0 + 1500.0
        return output
    
    @staticmethod
    def _pad_or_crop(x, target_h=70, target_w=70):
        """Pads or crops input tensor x to target height and width."""
        _, _, h, w = x.shape
        # Pad Height if needed
        if h < target_h:
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            x = F.pad(x, (0, 0, pad_top, pad_bottom))  # Pad height only
            h = target_h
        # Pad Width if needed
        if w < target_w:
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            x = F.pad(x, (pad_left, pad_right, 0, 0))  # Pad width only
            w = target_w
        # Crop Height if needed
        if h > target_h:
            crop_top = (h - target_h) // 2
            # Use slicing to crop
            x = x[:, :, crop_top : crop_top + target_h, :]
            h = target_h
        # Crop Width if needed
        if w > target_w:
            crop_left = (w - target_w) // 2
            x = x[:, :, :, crop_left : crop_left + target_w]
            w = target_w
        return x
