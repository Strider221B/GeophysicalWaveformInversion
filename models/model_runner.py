import csv
import gc
import glob
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from configs.config import Config
from helpers.constants import Constants
from helpers.helper import Helper
from helpers.kaggle_dataset import KaggleTestDataset
from models.model_factory import ModelFactory

class ModelRunner:

    @classmethod
    def train_model(cls,
                    dataloader_train: DataLoader,
                    dataloader_validation: DataLoader,
                    model: nn.Module,
                    optimizer: Optimizer,
                    loss_criterion) -> List[Dict[str, Any]]:
        history = []
        best_val_loss = float('inf')

        if dataloader_train is None or model is None:
            print("E: Training cannot proceed. Train DataLoader or Model is missing.")
            return
        try:
            for epoch in range(1, Config.n_epochs + 1):
                best_val_loss = cls._train_for(epoch,
                                               model,
                                               dataloader_train,
                                               dataloader_validation,
                                               optimizer,
                                               loss_criterion,
                                               history,
                                               best_val_loss)

        except KeyboardInterrupt:
            print("\n--- Training interrupted by user ---")
        except Exception as e:
            print(f"\nE: Training loop encountered a critical error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n--- Training Loop Finished ---")
        return history
    
    @classmethod
    def predict_on_kaggle_test_data(cls):
        print("\n --- Final Prediction on Kaggle Test Set ---")
        best_model_final_path = Helper.find_best_model()
        if not best_model_final_path:
            print("W: No best model found. Skipping final prediction.")
        elif not Path(Config.kaggle_test_dir).is_dir():
            print(f"W: Kaggle test directory '{Config.kaggle_test_dir}' not found. Skipping prediction.")
        else:
            cls._predict_on_kaggle_test_data_using(best_model_final_path)

        print("\n--- Full Workflow Finished ---")

    @classmethod
    def _predict_on_kaggle_test_data_using(cls, best_model_final_path: str):
        try:
            print(f"Loading model for final prediction: {os.path.basename(best_model_final_path)}")
            model_pred = ModelFactory.initialize_just_model()
            model_pred.load_state_dict(torch.load(best_model_final_path, map_location=Config.device))
            model_pred.eval()

            test_dataset = KaggleTestDataset(Config.kaggle_test_dir)
            if len(test_dataset) == 0:
                print("W: Kaggle test dataset is empty. No submission generated.")
                return
            # Setup DataLoader for test set
            # Use slightly smaller batch size and fewer workers for inference if needed
            test_batch_size = max(1, Config.batch_size // 2)
            test_num_workerd = min(max(0, Config.num_workers // 2), 
                                   (os.cpu_count() // 2 if os.cpu_count() else 1))
            dataloader_test = DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=test_num_workerd,
                pin_memory=Config.use_cuda,
            )
            print(f"Test DataLoader created with bs={test_batch_size}, workers={test_num_workerd}")
            print(f"Writing submission file to: {Config.submission_file}")

            rows_written = 0
            with open(Config.submission_file, "wt", newline="") as csvfile:
                cls._write_submission_file(csvfile, dataloader_test, model_pred)

            print(f"Submission file created: {Config.submission_file} ({rows_written} rows).")
            # Sanity check row count
            expected_rows = len(test_dataset) * 70  # 70 y-positions per test sample
            if rows_written != expected_rows:
                print(
                    f"W: Row count mismatch! Expected {expected_rows}, but wrote {rows_written}."
                )

        except Exception as e:
            print(f"E: Final prediction process failed critically: {e}")
            import traceback
            traceback.print_exc()

    @classmethod
    def _write_submission_file(cls, 
                               csvfile, 
                               dataloader_test: DataLoader,
                               model_pred: nn.Module):
        # Define CSV header columns (x_1, x_3, ..., x_69)
        x_cols = [f"x_{i}" for i in range(1, 70, 2)]
        fieldnames = ["oid_ypos"] + x_cols
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        pbar_test = tqdm(dataloader_test, desc="Generating Submission", unit="batch")
        with torch.no_grad():
            for inputs, original_ids in pbar_test:
                cls._write_for_batch(model_pred, x_cols, writer, original_ids)

    @classmethod
    def _write_for_batch(cls,
                         model_pred: nn.Module,
                         x_cols: List[str],
                         writer: csv.DictWriter,
                         original_ids: List[str]):
        # Handle batch size = 1 where original_ids might be a string
        if isinstance(original_ids, str):
            original_ids = [original_ids]
        try:
            inputs = inputs.to(Config.device).float()
            with torch.amp.autocast(
                device_type=Config.device.type,
                dtype=Config.autocast_dtype,
                enabled=Config.use_cuda,
            ):
                outputs = model_pred(inputs)
            # Output shape is (B, 1, H, W), get predictions (B, H, W)
            preds = outputs[:, 0].cpu().numpy()

            # Iterate through samples in the batch
            for y_pred, oid in zip(preds, original_ids): # y_pred is (H, W)
                # Iterate through y-positions (rows) for this sample
                for y_pos in range(y_pred.shape[0]): # y_pred.shape[0] should be 70
                    # Extract values at odd x-indices (1, 3, ..., 69)
                    vals = y_pred[y_pos, 1::2].astype(np.float32)
                    # Create row dictionary
                    row = dict(zip(x_cols, vals))
                    row["oid_ypos"] = f"{oid}_y_{y_pos}"
                    writer.writerow(row)
                    rows_written += 1
        except Exception as e:
            # Report error but continue if possible
            print(
                f"\nE: Prediction failed for batch (OID: {original_ids[0] if original_ids else '?'}) : {e}"
            )

    @classmethod
    def _train_for(cls,
                   epoch: int,
                   model: nn.Module,
                   dataloader_train: DataLoader,
                   dataloader_validation: DataLoader,
                   optimizer: Optimizer,
                   loss_criterion,
                   history: List[Dict[str, Any]],
                   best_val_loss: float):
        print(f"\n=== Epoch {epoch}/{Config.n_epochs} ===")
        # --- Training Phase ---
        gc.collect()
        if Config.use_cuda:
            torch.cuda.empty_cache()
        model.train()
        train_losses = []
        progess_bar_train = tqdm(dataloader_train, desc=f"Train E{epoch}", leave=False, unit="batch")
        for i, batch in enumerate(progess_bar_train):
            cls._train_batch(batch, i, optimizer, model, loss_criterion, train_losses, progess_bar_train)

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        print(f"Epoch {epoch} Avg Train Loss: {avg_train_loss:.5f}")

        # --- Validation Phase ---
        if dataloader_validation is None:
            print("W: Skipping validation phase - no validation DataLoader.")
            history.append({"epoch": epoch, "train_loss": avg_train_loss, "valid_loss": None})
            return  # Skip to next epoch

        model.eval()
        val_losses = []
        pbar_val = tqdm(dataloader_validation, desc=f"Valid E{epoch}", leave=False, unit="batch")
        with torch.no_grad():
            for i, batch in enumerate(pbar_val):
                cls._run_validation_batch(batch, model, loss_criterion, val_losses, i, epoch)

        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
        print(f"Epoch {epoch} Avg Valid Loss: {avg_val_loss:.5f}")
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "valid_loss": avg_val_loss})

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            cls._save_model(best_val_loss, epoch, model)

    @classmethod
    def _save_model(cls,
                    best_val_loss: float,
                    epoch: int,
                    model: nn.Module):
        cls._remove_old_best_model()
        # Save the new best model
        fname = f"{Config.model_prefix}_epoch_{epoch}_loss_{best_val_loss:.4f}{Constants.EXTN_MODEL}"
        fpath = os.path.join(Config.working_dir, fname)
        print(f"*** New best validation loss: {best_val_loss:.5f}. Saving model: {fname} ***")
        torch.save(model.state_dict(), fpath)

    @staticmethod
    def _remove_old_best_model():
        # Clean previous best models before saving new one
        del_pattern = os.path.join(
            Config.working_dir, f"{Config.model_prefix}_epoch_*_loss_*{Constants.EXTN_MODEL}"
        )
        for old_model_path in glob.glob(del_pattern):
            try:
                print(f"Removing model: {os.path.basename(old_model_path)}")
                os.remove(old_model_path)
            except OSError as e:
                print(f"W: Could not delete old model {old_model_path}: {e}")

    @classmethod
    def _run_validation_batch(cls,
                              batch: dict,
                              model: nn.Module,
                              loss_criterion,
                              val_losses: List[float],
                              batch_index: int,
                              epoch: int):
        if cls._is_batch_valid(batch) == False:
            return
        try:
            inputs = batch["seis"].to(Config.device, non_blocking=True).float()
            targets = batch["vel"].to(Config.device, non_blocking=True).float()
            with torch.amp.autocast(
                device_type=Config.device.type,
                dtype=Config.autocast_dtype,
                enabled=Config.use_cuda,
            ):
                outputs = model(inputs)
                loss = loss_criterion(outputs, targets)
            val_losses.append(loss.item())

            # Plotting validation examples periodically
            if batch_index == 0 and epoch % Config.plot_every_n_epochs == 0:
                # Add validation plotting code here if desired
                pass # Placeholder

        except Exception as e:
            print(f"\nE: Validation batch {batch_index} failed: {e}")
            if isinstance(e, torch.cuda.OutOfMemoryError):
                print("E: CUDA Out of Memory during validation. Exiting.")
                raise e
            # Continue on other errors if desired, or raise
            # raise e # Uncomment to stop on any validation error

    @classmethod
    def _train_batch(cls,
                     batch: dict, 
                     batch_index: int,
                     optimizer: Optimizer,
                     model: nn.Module,
                     loss_criterion,
                     train_losses: List[float],
                     pbar_train: tqdm):
        if cls._is_batch_valid(batch) == False:
            return
        try:
            inputs = batch["seis"].to(Config.device, non_blocking=True).float()
            targets = batch["vel"].to(Config.device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            # Use Automatic Mixed Precision (AMP) if on CUDA
            with torch.amp.autocast(
                device_type=Config.device.type,
                dtype=Config.autocast_dtype,
                enabled=Config.use_cuda,
            ):
                outputs = model(inputs)
                loss = loss_criterion(outputs, targets)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Update progress bar description
            if batch_index % 100 == 0:
                pbar_train.set_postfix(loss=f"{np.mean(train_losses):.5f}")

        except Exception as e:
            print(f"\nE: Training batch {batch_index} failed: {e}")
            # Stop training on OOM error
            if isinstance(e, torch.cuda.OutOfMemoryError):
                print("E: CUDA Out of Memory during training. Exiting.")
                raise e
            # Continue on other errors if desired, or raise
            # raise e # Uncomment to stop on any training error

    @staticmethod
    def _is_batch_valid(batch: dict):
        if not batch or Constants.SEIS not in batch or Constants.VEL not in batch:
            return False
        return True
