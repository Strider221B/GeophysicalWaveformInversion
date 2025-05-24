import logging
import os
import torch

from configs.model_configs.base_model_config import BaseModelConfig
from configs.platform_configs.base_platform_config import BasePlatformConfig
from helpers.constants import Constants

class Config:

    working_dir = ''
    shard_output_dir = ''
    train_dir = ''
    test_dir = ''

    submission_file = os.path.join(working_dir, "submission.csv")
    gpu_local_rank = int(os.environ.get('RANK', 0))
    gpu_world_size = int(os.environ.get('WORLD_SIZE', gpu_local_rank + 1))

    # --- Dataset Params ---
    dataset_name = "fwi_kaggle_only_augmented"

    # --- Sharding Params ---
    maxsize = 1e9  # Approx 1 GB
    force_shard_creation = False

    # --- Splitting & Loading Params ---
    num_used_shards = None  # Use all available
    test_size = 0.1  # Proportion for validation split
    batch_size = 16
    num_workers = 2

    # --- Augmentation Params ---
    apply_augmentation = True
    aug_hflip_prob = 0.5  # Probability of horizontal flip
    aug_seis_noise_std = 0.01  # Std dev of Gaussian noise added to seismic data
    reciever_flip = 0.5

    # --- Model params ---
    model_prefix = None
    n_epochs = None
    learning_rate = None
    weight_decay = None
    plot_every_n_epochs = None

    # --- Misc ---
    seed = 42
    use_cuda = False
    device = None
    autocast_dtype = None
    log_level = logging.WARNING
    trial_run = False

    @classmethod
    def initialize_params_with(cls, model_config: BaseModelConfig, platform_config: BasePlatformConfig):
        cls.model_prefix = model_config.model_prefix
        cls.n_epochs = model_config.get_epochs(cls.trial_run)
        cls.learning_rate = model_config.learning_rate
        cls.weight_decay = model_config.weight_decay
        cls.plot_every_n_epochs = model_config.plot_every_n_epochs
        cls.use_cuda = model_config.use_cuda
        cls.device = model_config.device
        cls.autocast_dtype = model_config.autocast_dtype

        cls.working_dir = platform_config.working_dir
        cls.shard_output_dir = platform_config.shard_output_dir
        cls.train_dir = platform_config.train_dir
        cls.test_dir = platform_config.test_dir
