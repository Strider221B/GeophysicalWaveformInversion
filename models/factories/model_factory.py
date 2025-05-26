from typing import Tuple

from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from configs.config import Config
from configs.model_configs.hg_net_v2_config import HG_Net_V2_Config
from configs.model_configs.u_net_config import UNetConfig
from models.factories.factory_base import FactoryBase
from models.factories.hg_net_v2_factory import HGNetV2Factory
from models.factories.unet_factory import UnetFactory

class ModelFactory(FactoryBase):

    @classmethod
    def get_model_with_components(cls) -> Tuple[nn.Module, Optimizer, nn.L1Loss, LRScheduler]:
        return  cls._get_factory_class().get_model_with_components()

    @classmethod
    def get_just_model(cls) -> nn.Module:
        return  cls._get_factory_class().get_just_model()

    @staticmethod
    def _get_factory_class() -> FactoryBase:
        if Config.get_model_prefix() == UNetConfig.model_prefix:
            return UnetFactory
        if Config.get_model_prefix() == HG_Net_V2_Config.model_prefix:
            return HGNetV2Factory
        raise ValueError('Invalid model provided.')
