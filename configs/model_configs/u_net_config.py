from configs.model_configs.base_model_config import BaseModelConfig

class UNetConfig(BaseModelConfig):

    unet_in_channels = 5
    unet_out_channels = 1
    unet_init_features = 32
    unet_depth = 5
    unet_bilinear = True

    # --- Base Model Parameters ---
    model_prefix = 'unet_best_model'
    learning_rate = 1e-4
    weight_decay = 1e-5
    plot_every_n_epochs = 5
    scheduler_factor = 0.1 # default
    scheduler_patience = 10 # default

    _n_epochs = 100