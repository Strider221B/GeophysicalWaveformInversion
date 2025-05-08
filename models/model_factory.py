from typing import Tuple

import torch
from torch.optim.optimizer import Optimizer
from torch import nn

from configs.config import Config
from configs.model_configs.u_net_config import UNetConfig
from models.u_net.unet import UNet

class ModelFactory:

    @classmethod
    def initialize_model_with_components(cls):
        if Config.model_prefix == UNetConfig.model_prefix:
            return cls._initialize_unet()
        raise ValueError('Invalid model provided.')

    @staticmethod
    def initialize_just_model():
        if Config.model_prefix == UNetConfig.model_prefix:
            return UNet().to(Config.device)
        raise ValueError('Invalid model provided.')

    @classmethod
    def _initialize_unet(cls) -> Tuple[nn.Module, Optimizer, nn.L1Loss]:
        model = None
        optimizer = None
        criterion = None
        try:
            model = cls.initialize_just_model()
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model: {model.__class__.__name__}, Trainable Params: {params:,}")
            criterion = nn.L1Loss()  # Mean Absolute Error
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=Config.learning_rate,
                                          weight_decay=Config.weight_decay)
            print(f"Loss Function: {criterion.__class__.__name__}")
            print(f"Optimizer: {optimizer.__class__.__name__} "
                  f"(lr={Config.learning_rate}, wd={Config.weight_decay})")
        except Exception as e:
            print(f"E: Model initialization failed: {e}")
            raise
        return model, optimizer, criterion
