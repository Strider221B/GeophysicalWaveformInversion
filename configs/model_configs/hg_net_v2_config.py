from configs.model_configs.base_model_config import BaseModelConfig

class HG_Net_V2_Config(BaseModelConfig):

    # --- Base Model Parameters ---
    model_prefix = 'hg_net_v2'
    n_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-3
    plot_every_n_epochs = 5
    scheduler_factor = 0.8
    scheduler_patience = 0
