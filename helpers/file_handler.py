import glob
import gc
import os
import shutil
from pathlib import Path

from config import Config
from helpers.constants import Constants

class FileHandler:

    @classmethod
    def clean_up(cls):
        paths_to_clean = [Config.shard_output_dir]
        # Find previous best model files based on pattern
        model_pattern = os.path.join(Config.working_dir, f"{Config.model_prefix}_epoch_*_loss_*.pth")
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
