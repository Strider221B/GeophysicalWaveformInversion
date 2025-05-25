import timm
import torch
from torch import nn

from helpers.helper import Helper
from models.hg_net_v2.segmentation_head_2d import SegmentationHead2d
from models.hg_net_v2.unet_decoder_2d import UnetDecoder2d

class HGUNet(nn.Module):

    _NUMBER_OF_CHANNELS = 'num_chs'

    def __init__(
        self,
        backbone: str,
    ):
        super().__init__()

        # Encoder
        self._backbone = timm.create_model(backbone, in_chans=5, pretrained=True, features_only=True, drop_path_rate= 0.2)
        encoder_channels = [features[self._NUMBER_OF_CHANNELS] for features in self._backbone.feature_info][::-1]
        decoder_channels = (256, 128, 64, 32)

        # Decoder
        self._decoder = UnetDecoder2d(encoder_channels=encoder_channels,
                                      decoder_channels=decoder_channels,
                                      scale_factors=(1,2,2,2))

        self._seg_head = SegmentationHead2d(in_channels=decoder_channels[-1],
                                            out_channels=1,
                                            scale_factor=2)
        self._update_stem()

    def forward(self, x: torch.Tensor):
        x = Helper.compress_sample(x) # Updates shape from (16, 5, 1000, 70) to (16, 5, 72, 72)
        x_in = x
        x_seg = self._forward_steps(x)

        if self.training:
            return x_seg
        p1 = self._proc_flip(x_in)
        x_seg = torch.mean(torch.stack([x_seg, p1]), dim=0)
        return x_seg

    def _update_stem(self, ):
        self._backbone.stem.stem1.conv.stride=(1,1)
        self._backbone.stages_3.downsample.conv.stride=(1,1)

    def _proc_flip(self, x_in):
        # Flip TTA during inference
        x_in = torch.flip(x_in, dims=[-3, -1])
        x_seg = self._forward_steps(x_in)
        x_seg = torch.flip(x_seg, dims=[-1])
        return x_seg

    def _forward_steps(self, x: torch.Tensor):
        x = self._backbone(x)
        x = x[::-1]
        x = self._decoder(x)
        x_seg = self._seg_head(x[-1])
        x_seg = x_seg[..., 1:-1, 1:-1]
        x_seg = x_seg * 1500 + 3000
        return x_seg
