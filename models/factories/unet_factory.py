from typing import Tuple

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from configs.config import Config
from configs.model_configs.u_net_config import UNetConfig
from models.factories.factory_base import FactoryBase
from models.u_net.unet import UNet

class UnetFactory(FactoryBase):

    @classmethod
    def get_model_with_components(cls) -> Tuple[nn.Module, Optimizer, nn.L1Loss, LRScheduler]:
        model = None
        optimizer = None
        criterion = None
        scheduler = None
        try:
            model = cls.get_just_model()
            params = cls._count_params(model)
            print(f"Model: {model.__class__.__name__}, Trainable Params: {params:,}")
            criterion = nn.L1Loss()  # Mean Absolute Error
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=UNetConfig.learning_rate,
                                          weight_decay=UNetConfig.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer,
                                          cls._SCHEDULER_MODE,
                                          UNetConfig.scheduler_factor,
                                          UNetConfig.scheduler_patience)
        except Exception as e:
            print(f"E: Model initialization failed: {e}")
            raise
        return model, optimizer, criterion, scheduler

    @staticmethod
    def get_just_model() -> UNet:
        return UNet().to(Config.get_device())
