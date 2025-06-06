import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
import webdataset as wds
from sklearn.model_selection import train_test_split

from configs.config import Config
from helpers.constants import Constants
from helpers.logger import Logger

class WebdatasetPreprocessing:

    _BUFFER_SIZE_FOR_SHUFFLING = 1000
    _KEY = "__key__"
    _NPY_SEIS = 'seis.npy'
    _NPY_VEL = 'vel.npy'
    _TXT_SAMPLE_ID = 'sample_id.txt'

    _logger = Logger.get_logger()

    @classmethod
    def search_data_path(cls,
                         target_dirs: List[str],
                         root_dir: str,
                         shuffle: bool=True,
                         seed:int=42) -> List[Tuple[Path, Path]]:
        """Finds input/output .npy file pairs within subdirectories of a root directory."""
        result_files = []
        root_path = Path(root_dir)
        if not root_path.is_dir():
            print(f"W: Root directory not found: {root_path}")
            return []

        print(f"Searching for data families {target_dirs} in root: {root_path}")
        total_pairs_found = 0
        for target_dir in target_dirs:
            total_pairs_found += cls._search_for_data_families_using(root_path, target_dir, result_files)

        print(f"Found {len(result_files)} total valid pairs across specified families.")
        if shuffle and result_files:
            print(f"Shuffling {len(result_files)} pairs (seed={seed}).")
            rng = np.random.default_rng(seed)
            rng.shuffle(result_files)

        return result_files

    @classmethod
    def generate_sample(cls, in_file: Path, out_file: Path=None, base_dir=None):
        """
        Loads data from .npy files, prepares dicts for WebDataset, converts to float16.
        Handles errors during loading gracefully.
        """
        data = []
        seis = None  # Initialize to ensure variable exists for potential del
        vel = None
        try:
            if out_file is None:
                # Logic for test data sharding (if needed later) - not implemented here
                print("W: generate_sample called without out_file (test mode?), not implemented.")
                return []
            # --- Load Train/Validation data ---
            try:
                seis, vel = cls._load_train_and_val_data(in_file, out_file)
            except:
                return []

            n_samples = cls._get_number_of_samples(seis, vel, in_file.name, out_file.name)

            if n_samples == 0:
                print(f"W: Found 0 samples in pair: {in_file.name}, {out_file.name}")
                del seis, vel
                return []

            # --- Generate unique key based on file path relative to base_dir ---
            unique_key = f"{in_file.parent.name}_{in_file.stem}"  # Default key
            if base_dir:
                key_using_base_dir = cls._create_key_from_relative_path(in_file, base_dir)
                unique_key = key_using_base_dir if len(key_using_base_dir) > 0 else unique_key

            # --- Process and append each sample ---
            for i in range(n_samples):
                cls._extract_and_convert_to_float16(unique_key, i, seis, vel, data)

            # --- Explicitly delete mmap objects after copying data ---
            # This is important to release file handles, especially with mmap
            del seis
            del vel

        except Exception as e:
            # Catch other errors (ValueError from dim check, key gen errors, etc.)
            print(f"E: Error during sample generation for {in_file.name}: {e}")
            # Explicitly try deleting here, in case they were loaded before the error
            if seis is not None:
                del seis
            if vel is not None:
                del vel
            return []  # Return empty list on any error during processing

        # No finally block needed as del is handled within try/except scopes
        return data

    @classmethod
    def get_shard_paths(cls,
                        root_dir: str,
                        dataset_name: str,
                        stage,
                        num_shards=None,
                        test_size=0.2, seed=42):
        """Gets list of shard paths, optionally selects subset, optionally splits train/val."""
        source_dir_name = f"train_{dataset_name}"
        dataset_dir = Path(root_dir) / source_dir_name
        print(f"Looking for shards for stage '{stage}' in: {dataset_dir}")

        if not dataset_dir.is_dir():
            print(f"W: Shard directory not found: {dataset_dir}")
            return (None, None) if stage == Constants.TRAIN else None

        shard_paths = sorted([str(p) for p in dataset_dir.glob(f"*{Constants.EXTN_SHARD}")])

        if not shard_paths:
            print(f"W: No {Constants.EXTN_SHARD} shards found in {dataset_dir}.")
            return (None, None) if stage == Constants.TRAIN else None

        print(f"Found {len(shard_paths)} total shards.")

        # --- Shard Selection Logic ---
        selected_paths = cls._subselect_required_number_of_shards(shard_paths, num_shards, seed)
        print(f"Using {len(selected_paths)} selected shards for stage '{stage}'.")

        # --- Train/Validation Split Logic ---
        if stage == Constants.TRAIN:
            return cls._return_train_and_val_path(selected_paths, test_size, seed)
        print(f"# Shards returned for stage '{stage}': {len(selected_paths)}")
        return sorted(selected_paths)

    @classmethod
    def get_dataset(cls,
                    paths: List[str],
                    stage: str,
                    seed: int=42) -> wds.WebDataset:
        """Creates WebDataset object. Applies augmentations if stage=='train'."""
        if not paths:
            print(f"W: No shard paths provided for stage '{stage}'. Cannot create dataset.")
            return None

        cls._update_paths_if_insufficient_for_multi_gpus(paths)

        print(f"Creating WebDataset for stage '{stage}' from {len(paths)} shards.")
        is_train = stage == Constants.TRAIN
        # Continue pipeline even if some samples fail decoding/mapping
        map_handler = wds.warn_and_continue

        try:
            dataset = wds.WebDataset(
                paths, nodesplitter=wds.split_by_node, shardshuffle=is_train, seed=seed
            )
            # Decode standard types (.npy, .txt, etc.)
            dataset = dataset.decode(handler=map_handler)

            # Apply the mapping function to train/val stages
            def map_train_val(sample: dict):
                return cls._map_train_val(is_train, sample)
            if stage in [Constants.TRAIN, Constants.VAL]:
                dataset = dataset.map(map_train_val, handler=map_handler)

            # Shuffle buffer for training data
            if is_train:
                dataset = dataset.shuffle(cls._BUFFER_SIZE_FOR_SHUFFLING)

            return dataset

        except Exception as e:
            print(f"E: Error creating WebDataset pipeline for stage '{stage}': {e}")
            return None

    @classmethod
    def _update_paths_if_insufficient_for_multi_gpus(cls, paths: List[str]):
        if Config.get_use_multiple_gpus():
            number_of_gpus = Config.get_gpu_world_size()
            path_len = len(paths)
            if (path_len > number_of_gpus) and (path_len % number_of_gpus == 0):
                return
            cls._logger.warning(f'Length of paths: {path_len} is not sufficient for GPUs: {number_of_gpus}')
            paths_to_re_add = 1
            if (path_len < number_of_gpus):
                paths_to_re_add = number_of_gpus - path_len
            # If we are not using torch's distributed data loader, we have to ensure the we have sufficient data
            # for all the GPUs and in equal quantity. Else, training step hangs with some of the GPUs completing
            # an epoch first and the other doesn't.
            # We need to ensure both the GPUs get same number of batches.
            while (paths_to_re_add + path_len) % number_of_gpus != 0:
                paths_to_re_add += 1
            cls._logger.warning(f'Updating paths to add last entry back again. Number of additions: {paths_to_re_add}')
            for _ in range(paths_to_re_add):
                paths.append(paths[-1]) # Hack, so that we have sufficient paths for every GPU.

    @classmethod
    def _map_train_val(cls,
                       is_train: bool,
                       sample: dict):
        key_info = sample.get(cls._KEY, "N/A")  # For error reporting
        try:
            required = [cls._TXT_SAMPLE_ID, cls._NPY_SEIS, cls._NPY_VEL]
            if not all(k in sample for k in required):
                raise KeyError(f"Missing required keys in sample {key_info}")

            sid = sample[cls._TXT_SAMPLE_ID]
            # Ensure numpy arrays and convert to float32 tensors
            s_np = np.asarray(sample[cls._NPY_SEIS])
            v_np = np.asarray(sample[cls._NPY_VEL])
            seis_tensor = torch.from_numpy(s_np).float()
            vel_tensor = torch.from_numpy(v_np).float()

            if is_train and Config.apply_augmentation:
                seis_tensor, vel_tensor = cls._apply_augmentation(seis_tensor, vel_tensor)

            return {Constants.SAMPLE_ID: sid,
                    Constants.SEIS: seis_tensor,
                    Constants.VEL: vel_tensor}

        except Exception as map_e:
            cls._logger.exception(f"E: Map function failed for sample {key_info}: {map_e}")
            # Let the handler decide whether to skip or raise
            raise map_e

    @staticmethod
    def _apply_augmentation(seis_tensor: torch.Tensor,
                            vel_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Horizontal Flip
        if torch.rand(1).item() < Config.aug_hflip_prob:
            seis_tensor = TF.hflip(seis_tensor)
            vel_tensor = TF.hflip(vel_tensor)
        # 2. Receiver Flip
        if torch.rand(1).item() < Config.reciever_flip:
            seis_tensor = seis_tensor.flip(dims=(0, 2))
            vel_tensor = vel_tensor.flip(dims=(1, ))
        # 3. Add Gaussian Noise to Seismic Data
        if Config.aug_seis_noise_std > 0:
            noise = torch.randn_like(seis_tensor) * Config.aug_seis_noise_std
            seis_tensor.add_(noise)  # In-place addition
        return seis_tensor, vel_tensor

    @classmethod
    def _extract_and_convert_to_float16(cls,
                                        unique_key: str,
                                        index: int,
                                        seis: np.ndarray,
                                        vel: np.ndarray,
                                        data: List[Dict[str, Any]]) -> List[dict]:
        key = f"{unique_key}_{index}"
        # Extract sample, explicitly copy, and convert to float16
        s_sample = cls._convert_to_float_16(seis, index)
        v_sample = cls._convert_to_float_16(vel, index)
        data.append(
            {
                cls._KEY: key,
                cls._TXT_SAMPLE_ID: key,  # Store key as text too
                cls._NPY_SEIS: s_sample,
                cls._NPY_VEL: v_sample,
            }
        )
        return data

    @staticmethod
    def _convert_to_float_16(array: np.ndarray, index: int):
        return (array[index].copy().astype(np.float16)
                if array.ndim == 4
                else array.copy().astype(np.float16))

    @classmethod
    def _create_key_from_relative_path(cls, in_file: Path, base_dir: str) -> str:
        common_part = ''
        try:
            # Create key from relative path parts, removing .npy suffix
            relative_path = in_file.relative_to(base_dir)
            common_part = "_".join(relative_path.parts).replace(Constants.EXTN_NUMPY.removeprefix('*'), "")
            # Ensure compatibility across OS path separators
            common_part = common_part.replace(os.sep, "_").replace("\\", "_")
        except ValueError:
            # If relative_to fails (e.g., different drives), use the default key
            pass
        return common_part

    @staticmethod
    def _get_number_of_samples(seis: np.ndarray,
                               vel: np.ndarray,
                               in_file_name: str,
                               out_file_name: str) -> int:
        n_samples = 0
        if seis.ndim == 4 and vel.ndim == 4:  # Batch of samples (N, C, H, W)
            if seis.shape[0] != vel.shape[0]:
                print(
                    f"W: Batch size mismatch in {in_file_name} ({seis.shape[0]}) vs {out_file_name} ({vel.shape[0]})"
                )
                del seis, vel
                return []
            n_samples = seis.shape[0]
        elif seis.ndim == 3 and vel.ndim == 3:  # Single sample (C, H, W)
            n_samples = 1
        else:
            # Raise error for unexpected dimensions
            raise ValueError(
                f"Unexpected dims: seis {seis.shape}, vel {vel.shape} in {in_file_name}"
            )
        return n_samples

    @classmethod
    def _search_for_data_families_using(cls,
                                        root_path: Path,
                                        target_dir: str,
                                        result_files: List[Tuple[Path, Path]]) -> int:
        data_dir = root_path / target_dir
        if not data_dir.is_dir():
            # print(f"W: Target directory {target_dir} not found in {root_path}")
            return 0

        in_files, out_files = [], []
        data_subdir = data_dir / "data"
        model_subdir = data_dir / "model"

        # Check for HF structure first, then Kaggle structure
        if data_subdir.is_dir() and model_subdir.is_dir():
            in_files = sorted(data_subdir.glob(Constants.EXTN_NUMPY))
            out_files = sorted(model_subdir.glob(Constants.EXTN_NUMPY))
        else:
            in_files = sorted(data_dir.glob(f"seis{Constants.EXTN_NUMPY}"))
            out_files = sorted(data_dir.glob(f"vel{Constants.EXTN_NUMPY}"))

        if not in_files or len(in_files) != len(out_files):
            if in_files or out_files:  # Only warn if some files were found
                print(
                    f"W: Mismatch or missing files in {data_dir} (in:{len(in_files)}, out:{len(out_files)}). Skipping."
                )
            return 0

        current_pairs = list(zip(in_files, out_files))
        result_files.extend(current_pairs)
        return len(current_pairs)

    @staticmethod
    def _load_train_and_val_data(in_file: Path, out_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        try:
            # Use mmap_mode='r' for memory efficiency if files are large
            seis = np.load(in_file, mmap_mode="r")
        except Exception as e:
            print(f"E: Load fail for input {in_file.name}: {e}")
            raise

        try:
            vel = np.load(out_file, mmap_mode="r")
        except Exception as e:
            print(f"E: Load fail for output {out_file.name}: {e}")
            # Clean up the already loaded seis if vel loading fails
            if seis is not None:
                del seis
            raise
        return seis, vel

    @staticmethod
    def _return_train_and_val_path(selected_paths: List[str],
                                   test_size: float,
                                   seed: int) -> Tuple[List[str], List[str]]:
        count = len(selected_paths)
        print(f"Splitting {count} selected shards (test_size={test_size}, seed={seed})")
        try:
            if not (0 <= test_size < 1):
                raise ValueError("test_size must be in [0, 1)")
            if count <= 1 or test_size == 0:
                reason = "only 1 shard" if count <= 1 else "test_size is 0"
                print(f"W: Cannot split for validation ({reason}). Assigning all to train.")
                return sorted(selected_paths), []
            trn_paths, val_paths = train_test_split(
                selected_paths, test_size=test_size, random_state=seed, shuffle=True
            )
            trn_paths.sort()
            val_paths.sort()
            print(f"# Train shards: {len(trn_paths)}, # Val shards: {len(val_paths)}")
            return trn_paths, val_paths
        except Exception as e:
            print(f"E: Failed to split shards: {e}")
            return None, None

    @staticmethod
    def _subselect_required_number_of_shards(shard_paths: List[str],
                                             num_shards: int,
                                             seed: int) -> List[str]:
        selected_paths = shard_paths
        available_count = len(shard_paths)
        if num_shards is not None:
            if 0 < num_shards < available_count:
                print(f"Selecting {num_shards} shards randomly (seed={seed}).")
                rng = np.random.default_rng(seed)
                indices = rng.choice(available_count, size=num_shards, replace=False)
                selected_paths = sorted([shard_paths[i] for i in indices])
            elif num_shards >= available_count:
                print(
                    f"Requested {num_shards} or more shards, using all {available_count} available."
                )
            else:  # num_shards <= 0
                print(
                    f"W: Invalid num_shards ({num_shards}). Using all {available_count} shards."
                )
        return selected_paths
