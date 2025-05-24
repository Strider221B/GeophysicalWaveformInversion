from typing import Tuple

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from configs.config import Config
from configs.model_configs.hg_net_v2_config import HG_Net_V2_Config
from helpers.constants import Constants
from models.factories.factory_base import FactoryBase
from models.hg_net_v2.hg_unet import HGUNet

class HGNetV2Factory(FactoryBase):

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
                                          lr=HG_Net_V2_Config.learning_rate,
                                          weight_decay=HG_Net_V2_Config.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer,
                                          cls._SCHEDULER_MODE,
                                          HG_Net_V2_Config.scheduler_factor,
                                          HG_Net_V2_Config.scheduler_patience)

        except Exception as e:
            print(f"E: Model initialization failed: {e}")
            raise
        return model, optimizer, criterion, scheduler

    @staticmethod
    def get_just_model() -> nn.Module:
        model = HGUNet(HG_Net_V2_Config.backbone)
        if Config.device.type == Constants.CUDA:
            model = model.to(Config.gpu_local_rank)
        return model
