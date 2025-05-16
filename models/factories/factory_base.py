from abc import ABC, abstractmethod
from typing import Tuple

from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

class FactoryBase(ABC):

    _SCHEDULER_MODE = 'min'

    @classmethod
    @abstractmethod
    def get_model_with_components(cls) -> Tuple[nn.Module, Optimizer, nn.L1Loss, LRScheduler]:
        pass

    @staticmethod
    @abstractmethod
    def get_just_model() -> nn.Module:
        pass

    @staticmethod
    def _count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
