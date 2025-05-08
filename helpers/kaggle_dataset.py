from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from helpers.constants import Constants

class KaggleTestDataset(Dataset):
    """Loads the final Kaggle test set directly from individual .npy files."""

    def __init__(self, test_files_dir: str):
        self._test_files_dir = Path(test_files_dir)
        self._test_files = []
        try:
            if not self._test_files_dir.is_dir():
                raise FileNotFoundError(f"Kaggle test directory missing: {self._test_files_dir}")
            self._test_files = sorted(list(self._test_files_dir.glob(Constants.EXTN_NUMPY)))
            print(f"Found {len(self._test_files)} '{Constants.EXTN_NUMPY}' files in Kaggle test dir: {self._test_files_dir}")
            if not self._test_files:
                print(f"W: No {Constants.EXTN_NUMPY} files found in {self._test_files_dir}.")
        except Exception as e:
            print(f"E: Error accessing Kaggle test directory {self._test_files_dir}: {e}")

    def __len__(self):
        return len(self._test_files)

    def __getitem__(self, index) -> Tuple[np.ndarray, str]:
        if not self._test_files or index >= len(self._test_files):
            raise IndexError(f"Index {index} out of bounds for KaggleTestDataset ({len(self._test_files)} files).")
        test_file_path = self._test_files[index]
        try:
            # Load numpy array and convert to float32 tensor
            data = torch.from_numpy(np.load(test_file_path).astype(np.float32))
            # Get the original ID (filename without extension)
            original_id = test_file_path.stem
            return data, original_id
        except Exception as e:
            # Raise a more informative error if loading fails
            raise IOError(f"Error loading Kaggle test file: {test_file_path}") from e
