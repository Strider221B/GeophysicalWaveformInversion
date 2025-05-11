from typing import List

import torch
from torch import nn

class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self._models = nn.ModuleList(models).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = None

        for m in self._models:
            logits = m(x)

            if output is None:
                output = logits
            else:
                output += logits

        output /= len(self._models)
        return output
