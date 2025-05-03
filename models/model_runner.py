import torch
import torch.nn as nn

from configs.config import Config
from configs.model_configs.u_net_config import UNetConfig
from models.u_net.unet import UNet

class ModelRunner:

    @classmethod
    def initialize_model(cls):
        print("\n--- Initializing Model, Loss, Optimizer ---")
        model = None
        try:
            model = cls._get_model_based_on_config()
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

    @staticmethod
    def _get_model_based_on_config() -> nn.Module:
        if Config.model_prefix == UNetConfig.model_prefix:
            return UNet().to(Config.device)
        raise ValueError('Invalid model provided.')
