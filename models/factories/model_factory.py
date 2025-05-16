from typing import Tuple

from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from configs.config import Config
from configs.model_configs.u_net_config import UNetConfig
from models.factories.factory_base import FactoryBase
from models.factories.unet_factory import UnetFactory

class ModelFactory(FactoryBase):

    @classmethod
    def get_model_with_components(cls) -> Tuple[nn.Module, Optimizer, nn.L1Loss, LRScheduler]:
        if Config.model_prefix == UNetConfig.model_prefix:
            return UnetFactory.get_model_with_components()
        raise ValueError('Invalid model provided.')

    @staticmethod
    def get_just_model() -> nn.Module:
        if Config.model_prefix == UNetConfig.model_prefix:
            return UnetFactory.get_just_model()
        raise ValueError('Invalid model provided.')
