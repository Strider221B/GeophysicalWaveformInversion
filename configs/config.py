import os
import torch

from configs.model_configs.u_net_config import UNetConfig as model_config

class Config:
   
    kaggle_train_dir = "/kaggle/input/waveform-inversion/train_samples"
    kaggle_test_dir = "/kaggle/input/waveform-inversion/test"
    shard_output_dir = "/kaggle/working/sharded_data"
    working_dir = "/kaggle/working/"
    submission_file = os.path.join(working_dir, "submission.csv")

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

    # --- Model params (U-Net) ---
    model_prefix = model_config.model_prefix
    learning_rate = model_config.learning_rate
    weight_decay = model_config.weight_decay


    # --- Misc ---
    seed = 42
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    autocast_dtype = torch.float16 if use_cuda else torch.bfloat16
