from copy import deepcopy

import torch
from torch import nn

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        self._module = deepcopy(model)
        self._module.eval()
        self._decay = decay
        self._device = device
        if self._device is not None:
            self._module.to(device=device)

    def get_module(self) -> nn.Module:
        return self._module

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self._decay * e + (1. - self._decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def forward(self, _):
        raise ValueError('Not expecting forward call in EMA')

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self._module.state_dict().values(), model.state_dict().values()):
                if self._device is not None:
                    model_v = model_v.to(device=self._device)
                ema_v.copy_(update_fn(ema_v, model_v))
