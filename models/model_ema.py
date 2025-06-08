from copy import deepcopy

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        self._set_module_from(model)
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

    def _set_module_from(self, model):
        if type(model) is DistributedDataParallel:
            # Distributed data parallel is already adding a wrapper on top of model. If we don't extract it here,
            # then while exporting it finally, we will end up saving it with distributed data parallel.
            # We then won't be able to run inference without using distributed parallel. Trying to load the
            # weights directly will lead to an exception:
            # Missing key(s) in state_dict: "_backbone.stem.stem1.conv.weight"
            # Unexpected key(s) in state_dict: "module._backbone.stem.stem1.conv.weight"
            # You see, a 'module' gets added at every location.
            model = model.module
        self._module = deepcopy(model)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self._module.state_dict().values(), model.state_dict().values()):
                if self._device is not None:
                    model_v = model_v.to(device=self._device)
                ema_v.copy_(update_fn(ema_v, model_v))
