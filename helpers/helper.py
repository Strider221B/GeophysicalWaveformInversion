import glob
import os
import random
from typing import List

import numpy as np
import torch

from configs.config import Config
from helpers.constants import Constants

class Helper:

    @staticmethod
    def set_seed(seed=42):
        """Sets seed for reproducibility across libraries."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if Config.use_cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure reproducibility if desired, may impact performance
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Seed set to {seed}")

    @classmethod
    def find_best_model(cls, model_dir=Config.working_dir, model_prefix=Config.model_prefix):
        """
        Finds the best model file based on filename pattern (lowest loss).
        Falls back to most recently created/modified if pattern fails or doesn't exist.
        """
        best_model_path = None
        pattern = os.path.join(model_dir, f"{model_prefix}_epoch_*_loss_*{Constants.EXTN_MODEL}")
        model_files = glob.glob(pattern)

        if not model_files:
            print(f"W: No models matching pattern '{pattern}'. Looking for *{Constants.EXTN_MODEL}")
            best_model_path = cls._find_latest_model_with(model_dir, Constants.EXTN_MODEL)

        elif "loss" in os.path.basename(pattern):
            print('Trying to find best model using loss info')
            best_model_path = cls._parse_loss_from_file_name(model_files)

        else:
            print('Using latest model')
            best_model_path = cls._get_latest_file_from(model_files)

        if not best_model_path:
            print(f"W: No models found, final fallback. Looking for *{Constants.EXTN_MODEL}")
            best_model_path = cls._find_latest_model_with(model_dir, Constants.EXTN_MODEL)

        return best_model_path
    
    @classmethod
    def initialize(cls):
        cls.set_seed(Config.seed)
        print(f"Device: {Config.device}")
        print(f"Using PyTorch version: {torch.__version__}")
        if Config.use_cuda:
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    
    @classmethod
    def _find_latest_model_with(cls, model_dir: str, extn: str = Constants.EXTN_MODEL) -> str:
        all_pth_files = glob.glob(os.path.join(model_dir, f"*{extn}"))
        latest_model_path = None
        if all_pth_files:
            latest_model_path = cls._get_latest_file_from(all_pth_files)
        else:
            print(f"W: No {extn} models found in model directory.")
        return latest_model_path

    @classmethod
    def _parse_loss_from_file_name(cls, model_file_names: List[str]) -> str:
        parsed_models = []
        best_model_path = None
        for model_file_name in model_file_names:
            try:
                loss_str = model_file_name.split("_loss_")[-1].split(Constants.EXTN_MODEL)[0]
                loss = float(loss_str)
                parsed_models.append((loss, model_file_name))
            except (ValueError, IndexError, AttributeError):
                print(f"W: Couldn't parse loss from filename: {os.path.basename(model_file_name)}")

        if parsed_models:
            # Found models with parseable loss, sort by loss
            parsed_models.sort(key=lambda x: x[0])
            best_loss, best_model_path = parsed_models[0]
            print(f"Found best model by loss: {os.path.basename(best_model_path)} (Loss: {best_loss:.4f})")
        elif model_file_names:
            # Pattern matched, but loss couldn't be parsed from any filename
            print("W: Pattern matched but no losses parsed. Selecting most recently created.")
            best_model_path = cls._get_latest_file_from(model_file_names)
            if best_model_path:
                print(f"Using most recent creation time: {os.path.basename(best_model_path)}")
        return best_model_path

    @staticmethod
    def _get_latest_file_from(file_paths: List[str]) -> str:
        return  max(file_paths, key=os.path.getctime, default=None)
