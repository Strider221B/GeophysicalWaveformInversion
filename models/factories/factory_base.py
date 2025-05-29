from abc import ABC, abstractmethod
from typing import Tuple

from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from configs.config import Config
from helpers.constants import Constants

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

    @staticmethod
    def _configure_model(model: nn.Module) -> nn.Module:
        if Config.get_device().type == Constants.CUDA:
            model = model.to(Config.get_gpu_local_rank())
        if Config.get_use_multiple_gpus():
            model = DistributedDataParallel(model, device_ids=[Config.get_gpu_local_rank()])
        return model
