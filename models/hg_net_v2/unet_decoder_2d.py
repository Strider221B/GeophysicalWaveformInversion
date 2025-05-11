from typing import List, Tuple

import torch
from torch import nn

from models.hg_net_v2.decoder_block_2d import DecoderBlock2d

class UnetDecoder2d(nn.Module):

    def __init__(
        self,
        encoder_channels: Tuple[int],
        skip_channels: Tuple[int] = None,
        decoder_channels: Tuple[int] = (256, 128, 64, 32, 16),
        scale_factors: Tuple[int] = (1,2,2,2),
    ):
        super().__init__()

        if len(encoder_channels) == 4:
            decoder_channels = decoder_channels[1:]
        self._decoder_channels = decoder_channels

        if skip_channels is None:
            skip_channels = list(encoder_channels[1:]) + [0]

        # Build decoder blocks
        in_channels= [encoder_channels[0]] + list(decoder_channels[:-1])
        self._blocks = nn.ModuleList()

        for ic, sc, dc, sf in zip(in_channels, skip_channels, decoder_channels, scale_factors):
            self._blocks.append(DecoderBlock2d(ic, sc, dc, scale_factor= sf))

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        res= [feats[0]]
        feats= feats[1:]

        # Decoder blocks
        for i, b in enumerate(self._blocks):
            skip = feats[i] if i < len(feats) else None
            res.append(b(res[-1], skip=skip))

        return res
