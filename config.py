import os
import torch

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
    unet_in_channels = 5
    unet_out_channels = 1
    unet_init_features = 32
    unet_depth = 5
    unet_bilinear = True

    # --- Training params ---
    n_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    plot_every_n_epochs = 5

    # --- Misc ---
    seed = 42
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    autocast_dtype = torch.float16 if use_cuda else torch.bfloat16