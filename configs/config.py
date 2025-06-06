import logging
import os

import torch

from configs.model_configs.base_model_config import BaseModelConfig
from configs.platform_configs.base_platform_config import BasePlatformConfig

class Config:

    # --- Dataset Params ---
    dataset_name = "fwi_kaggle_only_augmented"

    # --- Sharding Params ---
    maxsize = 1e9  # Approx 1 GB
    force_shard_creation = False

    # --- Splitting & Loading Params ---
    num_used_shards = None  # Use all available
    test_size = 0.1  # Proportion for validation split
    batch_size = 16

    # --- Augmentation Params ---
    apply_augmentation = True
    aug_hflip_prob = 0.5  # Probability of horizontal flip
    aug_seis_noise_std = 0.01  # Std dev of Gaussian noise added to seismic data
    reciever_flip = 0.5

    # --- Misc ---
    seed = 42
    log_level = logging.DEBUG
    trial_run = False

    early_stopping_epoch_count = 15

    @classmethod
    def initialize_params_with(cls, model_config: BaseModelConfig,
                               platform_config: BasePlatformConfig,
                               use_multiple_gpus: bool):
        cls._initialize_model_config(model_config)
        cls._initialize_platform_config(platform_config)
        cls.initialize_gpu_config(use_multiple_gpus)

    # ======= GPU config =======
    @classmethod
    def get_use_multiple_gpus(cls) -> bool:
        return cls._use_multiple_gpus

    @classmethod
    def get_gpu_local_rank(cls) -> int:
        return cls._gpu_local_rank

    @classmethod
    def get_gpu_world_size(cls) -> int:
        return cls._gpu_world_size

    @classmethod
    def get_multi_gpu_backend(cls) -> str:
        return cls._multi_gpu_backend

    @classmethod
    def get_num_workers(cls) -> int:
        return cls._get_num_workers

    # ======= Model config =======
    @classmethod
    def get_model_prefix(cls) -> str:
        return cls._model_prefix

    @classmethod
    def get_n_epochs(cls) -> int:
        return cls._n_epochs

    @classmethod
    def get_learning_rate(cls) -> float:
        return cls._learning_rate

    @classmethod
    def get_weight_decay(cls) -> float:
        return cls._weight_decay

    @classmethod
    def get_plot_every_n_epochs(cls) -> int:
        return cls._plot_every_n_epochs

    @classmethod
    def get_use_cuda(cls) -> bool:
        return cls._use_cuda

    @classmethod
    def get_device(cls) -> torch.device:
        return cls._device

    @classmethod
    def get_autocast_dtype(cls) -> torch.dtype:
        return cls._autocast_dtype

    # ======== Platform Config ==========
    @classmethod
    def get_working_dir(cls) -> str:
        return cls._working_dir

    @classmethod
    def get_shard_output_dir(cls) -> str:
        return cls._shard_output_dir

    @classmethod
    def get_train_dir(cls) -> str:
        return cls._train_dir

    @classmethod
    def get_test_dir(cls) -> str:
        return cls._test_dir

    @classmethod
    def get_submission_file(cls) -> str:
        return cls._submission_file

    @classmethod
    def initialize_gpu_config(cls, use_multiple_gpus: bool):
        cls._use_multiple_gpus = use_multiple_gpus
        if use_multiple_gpus:
            cls._gpu_local_rank = int(os.environ['RANK'])
            cls._multi_gpu_backend = 'nccl'
            cls._get_num_workers = 1
        else:
            cls._gpu_local_rank = 0
            cls._get_num_workers = 2
        cls._gpu_world_size = int(os.environ.get('WORLD_SIZE', cls._gpu_local_rank + 1))

    @classmethod
    def _initialize_model_config(cls, model_config: BaseModelConfig):
        cls._model_prefix = model_config.model_prefix
        cls._n_epochs = model_config.get_epochs(cls.trial_run)
        cls._learning_rate = model_config.learning_rate
        cls._weight_decay = model_config.weight_decay
        cls._plot_every_n_epochs = model_config.plot_every_n_epochs
        cls._use_cuda = model_config.use_cuda
        cls._device = model_config.device
        cls._autocast_dtype = model_config.autocast_dtype

    @classmethod
    def _initialize_platform_config(cls, platform_config: BasePlatformConfig):
        cls._working_dir = platform_config.working_dir
        cls._shard_output_dir = platform_config.shard_output_dir
        cls._train_dir = platform_config.train_dir
        cls._test_dir = platform_config.test_dir
        cls._submission_file = os.path.join(cls._working_dir, "submission.csv")
