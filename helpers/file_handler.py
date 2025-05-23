import glob
import gc
import os
import shutil
import traceback
from pathlib import Path
from typing import List, Tuple

from tqdm.auto import tqdm
import webdataset as wds
from torch.utils.data import DataLoader

from configs.config import Config
from helpers.constants import Constants
from helpers.webdataset_preprocessing import WebdatasetPreprocessing as wdp

class FileHandler:

    @staticmethod
    def clean_up():
        paths_to_clean = [Config.shard_output_dir]
        # Find previous best model files based on pattern
        model_pattern = os.path.join(Config.working_dir, f"{Config.model_prefix}_epoch_*_loss_*{Constants.EXTN_MODEL}")
        paths_to_clean.extend(glob.glob(model_pattern))
        # Add plot and submission files
        paths_to_clean.append(os.path.join(Config.working_dir, Constants.PNG_TRAINING_HISTORY))
        paths_to_clean.append(Config.submission_file)

        for path_str in paths_to_clean:
            path_obj = Path(path_str)
            try:
                if path_obj.is_dir():
                    shutil.rmtree(path_obj, ignore_errors=True)
                    print(f"Removed directory (if existed): {path_obj}")
                elif path_obj.is_file():
                    path_obj.unlink(missing_ok=True)  # Ignore error if file doesn't exist
                    print(f"Removed file (if existed): {path_obj}")
            except Exception as e:
                print(f"W: Error during cleanup of {path_obj}: {e}")
        print("--- Cleanup finished ---")
        gc.collect()

    @classmethod
    def shard_from_kaggle_data(cls):
        shard_stage_dir = Path(Config.shard_output_dir) / f"train_{Config.dataset_name}"
        kaggle_train_root = Path(Config.train_dir)

        try:
            shards_need_creation = cls._does_shards_need_to_be_created(shard_stage_dir)
            # --- Ensure output directories exist & Check Disk Space ---
            cls._check_sufficient_disk_space()
            cls._create_output_dirs(shard_stage_dir)

            # --- Sharding Process ---
            if shards_need_creation:
                cls._create_shards(kaggle_train_root, shard_stage_dir)
            else:
                existing_shard_count = len(list(shard_stage_dir.glob(f"*{Constants.EXTN_SHARD}")))
                print(f"Using {existing_shard_count} existing shards.")

        except Exception as e:
            print(f"E: Kaggle-only sharding process failed critically: {e}")
            traceback.print_exc()
            raise

    @classmethod
    def create_data_loaders_from_shards(cls) -> Tuple[List[str], DataLoader, DataLoader]:
        print("\n--- 2. Creating DataLoaders from Shards ---")
        dataloader_train, dataloader_validation = None, None
        validation_paths = []  # Keep track of validation paths for potential later use
        try:
            trn_paths, val_paths = wdp.get_shard_paths(Config.shard_output_dir,
                                                       Config.dataset_name,
                                                       Constants.TRAIN,  # Request splitting
                                                       num_shards=Config.num_used_shards,
                                                       test_size=Config.test_size,
                                                       seed=Config.seed)
            validation_paths = val_paths  # Save the validation paths

            if trn_paths is None:
                raise RuntimeError("Failed to get or split shard paths for train/val.")

            # Check if any shards actually exist if paths were returned empty
            shard_check_dir = Path(Config.shard_output_dir) / f"train_{Config.dataset_name}"
            if not list(shard_check_dir.glob(f"*{Constants.EXTN_SHARD}")):
                raise RuntimeError(
                    f"No training shards selected AND no .tar files found in {shard_check_dir}."
                )

            print(f"Using {len(trn_paths)} shards for training.")
            if not val_paths:
                print("W: No shards assigned for validation. Validation will be skipped.")
            else:
                print(f"Using {len(val_paths)} shards for validation.")

            # Create WebDatasets (Augmentation applied in get_dataset for 'train')
            trn_ds = wdp.get_dataset(trn_paths, Constants.TRAIN, seed=Config.seed) if trn_paths else None
            val_ds = wdp.get_dataset(val_paths, Constants.VAL, seed=Config.seed + 1) if val_paths else None

            # Check if dataset creation failed unexpectedly
            if trn_ds is None and trn_paths:
                raise RuntimeError("Failed to create train WebDataset pipeline.")
            if val_ds is None and val_paths:
                # Only warn if validation dataset failed, training might still proceed
                print("W: Failed to create validation WebDataset pipeline.")

            # Create DataLoaders
            if trn_ds:
                dataloader_train = cls._create_dataloader_from(trn_paths, trn_ds)
            if val_ds:
                dataloader_validation = cls._create_dataloader_from(val_paths, val_ds)

            cls._perform_final_check_on_loaders(dataloader_train,
                                                dataloader_validation,
                                                trn_paths,
                                                val_paths)

        except Exception as e:
            print(f"E: DataLoader creation failed critically: {e}")
            traceback.print_exc()
            raise
        return validation_paths, dataloader_train, dataloader_validation

    @staticmethod
    def _perform_final_check_on_loaders(dataloader_train: DataLoader,
                                        dataloader_validation: DataLoader,
                                        trn_paths: List[str],
                                        val_paths: List[str]):
        # Final check (can sometimes trigger TypeError: 'IterableDataset' has no len())
        try:
            loaders_exist = bool(dataloader_train or dataloader_validation)
            if not loaders_exist and (trn_paths or val_paths):
                # Should not happen if datasets were created but loaders failed
                raise RuntimeError("Loaders missing despite dataset paths existing.")
            print("DataLoader(s) created successfully or skipped appropriately.")
        except TypeError as te:
            # Expected error for IterableDataset without explicit length
            if "has no len()" in str(te):
                print(f"W: Caught expected TypeError '{te}'. Assume DataLoaders are ok.")
            else:
                raise te  # Re-raise unexpected TypeError

    @staticmethod
    def _create_dataloader_from(paths: List[str], dataset: wds.WebDataset) -> DataLoader:
        n_trn_w = min(Config.num_workers, len(paths)) if paths else 0
        p_trn = n_trn_w > 0  # Use persistent workers only if num_workers > 0
        dataloader = DataLoader(dataset.batched(Config.batch_size),
                                batch_size=None,  # Already batched by WebDataset
                                shuffle=False,  # Shuffling done by WebDataset
                                num_workers=n_trn_w,
                                pin_memory=Config.use_cuda,
                                persistent_workers=p_trn,
                                prefetch_factor=2 if p_trn else None) # Only relevant if num_workers > 0
        print(f"DataLoader created with {n_trn_w} workers.")
        return dataloader

    @classmethod
    def _create_shards(cls,
                       kaggle_train_root: Path,
                       shard_stage_dir: Path):
        print(
            f"Starting shard creation from {kaggle_train_root} into {shard_stage_dir}"
        )
        if not kaggle_train_root.is_dir():
            raise FileNotFoundError(f"Kaggle train directory not found: {kaggle_train_root}")

        # Find family subdirectories in the Kaggle train directory
        families = [d.name for d in kaggle_train_root.iterdir() if d.is_dir()]
        print(f"Searching Kaggle data families: {families}")
        if not families:
            raise FileNotFoundError(
                f"No family subdirectories found in {kaggle_train_root}"
            )

        print("Searching for all data pairs in Kaggle source...")
        kaggle_file_pairs = wdp.search_data_path(
            families, kaggle_train_root, shuffle=True, seed=Config.seed
        )
        print(f"Found {len(kaggle_file_pairs)} total valid pairs from Kaggle source.")
        if not kaggle_file_pairs:
            raise RuntimeError(
                "No valid data pairs found in the specified Kaggle directories."
            )

        cls._write_shards(shard_stage_dir, kaggle_train_root, kaggle_file_pairs)

    @staticmethod
    def _write_shards(shard_stage_dir: Path,
                      kaggle_train_root: Path,
                      kaggle_file_pairs: List[Tuple[Path, Path]]):
        shard_pattern = str(shard_stage_dir / f"%06d{Constants.EXTN_SHARD}")
        print(
            f"Writing shards using pattern {shard_pattern} (max size {Config.maxsize / 1e9:.2f} GB)"
        )
        with wds.ShardWriter(shard_pattern, maxsize=int(Config.maxsize)) as writer:
            common_base_dir = kaggle_train_root  # For relative path key generation
            total_samples_written = 0
            count = 0
            for in_file, out_file in tqdm(
                kaggle_file_pairs, desc="Sharding Kaggle Data", unit="pair"
            ):
                count += 1
                # generate_sample handles potential errors for each pair
                samples_from_pair = wdp.generate_sample(
                    Path(in_file), Path(out_file), base_dir=common_base_dir
                )
                if samples_from_pair:
                    for sample_dict in samples_from_pair:
                        writer.write(sample_dict)
                    total_samples_written += len(samples_from_pair)
                if Config.trial_run and count > 4:
                    break

        print(
            f"Finished writing {total_samples_written} samples from Kaggle source to shards."
        )

    @staticmethod
    def _does_shards_need_to_be_created(shard_stage_dir: Path) -> bool:
        # --- Check if shards need creating ---
        needs_creation = True
        if shard_stage_dir.exists() and any(shard_stage_dir.glob(f"*{Constants.EXTN_SHARD}")):
            if Config.force_shard_creation:
                print(f"Forcing shard creation. Removing existing shards in {shard_stage_dir}")
                shutil.rmtree(shard_stage_dir)
            else:
                print(f"Found existing shards at: {shard_stage_dir}. Skipping creation.")
                needs_creation = False
        return needs_creation

    @staticmethod
    def _check_sufficient_disk_space():
        print("\n--- Checking Disk Space Before Directory Creation ---")
        try:
            total, used, free = shutil.disk_usage(Config.working_dir)
            print(
                f"Disk Usage for {Config.working_dir}: Total={total / 1e9:.2f}GB, Used={used / 1e9:.2f}GB, Free={free / 1e9:.2f}GB"
            )
        except Exception as du_e:
            print(f"W: Could not check disk usage: {du_e}")

    @staticmethod
    def _create_output_dirs(shard_stage_dir: Path):
        try:
            Path(Config.shard_output_dir).mkdir(parents=True, exist_ok=True)
            shard_stage_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"E: Critical error creating output directories: {e}")
            raise  # Stop if directories can't be created
