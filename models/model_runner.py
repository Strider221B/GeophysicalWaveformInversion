import csv
import gc
import glob
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm.auto import tqdm
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from configs.config import Config
from helpers.constants import Constants
from helpers.gpu_helper import GPUHelper
from helpers.helper import Helper
from helpers.logger import Logger
from helpers.kaggle_dataset import KaggleTestDataset
from models.factories.model_factory import ModelFactory
from models.model_ema import ModelEMA

class ModelRunner:

    _logger = Logger.get_logger()
    _MINIMUM_REQUIRED_IMPROVEMENT_PER_EPOCH = 1e-5

    @classmethod
    def train_model(cls,
                    dataloader_train: DataLoader,
                    dataloader_validation: DataLoader,
                    model: nn.Module,
                    optimizer: Optimizer,
                    loss_criterion,
                    scheduler: LRScheduler) -> List[Dict[str, Any]]:
        history = []
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        if dataloader_train is None or model is None:
            cls._logger.error("Training cannot proceed. Train DataLoader or Model is missing.")
            return history
        try:
            for epoch in range(1, Config.get_n_epochs() + 1):
                prev_best_val_loss = best_val_loss
                best_val_loss = cls._train_for(epoch,
                                               model,
                                               dataloader_train,
                                               dataloader_validation,
                                               optimizer,
                                               loss_criterion,
                                               scheduler,
                                               history,
                                               best_val_loss)
                if cls._is_early_stopping_required(best_val_loss, prev_best_val_loss):
                    break

        except KeyboardInterrupt:
            cls._logger.warning("--- Training interrupted by user ---")
        except Exception as e:
            cls._logger.exception(f"Training loop encountered a critical error: {e}")
        finally:
            cls._logger.info("--- Training Loop Finished ---")
        return history

    @classmethod
    def predict_on_kaggle_test_data(cls):
        cls._logger.info("--- Final Prediction on Kaggle Test Set ---")
        best_model_final_path = Helper.find_best_model_path(Config.get_working_dir(), Config.get_model_prefix())
        if not best_model_final_path:
            cls._logger.warning("No best model found. Skipping final prediction.")
        elif not Path(Config.get_test_dir()).is_dir():
            cls._logger.warning(f"Kaggle test directory '{Config.get_test_dir()}' not found. Skipping prediction.")
        else:
            cls._predict_on_kaggle_test_data_using(best_model_final_path)

        cls._logger.info("\n--- Full Workflow Finished ---")

    @classmethod
    def _predict_on_kaggle_test_data_using(cls, best_model_final_path: str):
        try:
            cls._logger.info(f"Loading model for final prediction: {os.path.basename(best_model_final_path)}")
            model_pred = ModelFactory.get_just_model()
            model_pred.load_state_dict(torch.load(best_model_final_path, map_location=Config.get_device()))
            model_pred.eval()

            test_dataset = KaggleTestDataset(Config.get_test_dir())
            if len(test_dataset) == 0:
                cls._logger.warning("Kaggle test dataset is empty. No submission generated.")
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
                pin_memory=Config.get_use_cuda(),
            )
            cls._logger.info(f"Test DataLoader created with bs={test_batch_size}, workers={test_num_workerd}")
            cls._logger.info(f"Writing submission file to: {Config.get_submission_file()}")

            with open(Config.get_submission_file(), "wt", newline="") as csvfile:
                rows_written = cls._write_submission_file(csvfile, dataloader_test, model_pred)

            cls._logger.info(f"Submission file created: {Config.get_submission_file()} ({rows_written} rows).")
            # Sanity check row count
            expected_rows = len(test_dataset) * 70  # 70 y-positions per test sample
            if rows_written != expected_rows:
                cls._logger.warning(
                    f"Row count mismatch! Expected {expected_rows}, but wrote {rows_written}."
                )

        except Exception as e:
            cls._logger.exception(f"Final prediction process failed critically: {e}")

    @classmethod
    def _is_early_stopping_required(cls,
                                    current_best_val_loss: float,
                                    prev_best_val_loss: float) -> bool:
        if (current_best_val_loss - prev_best_val_loss) < cls._MINIMUM_REQUIRED_IMPROVEMENT_PER_EPOCH:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0
        if epochs_without_improvement >= Config.early_stopping_epoch_count:
            cls._logger.warning(f'Early stopping criteria achived. '
                                f'Waited for: {Config.early_stopping_epoch_count}. '
                                f'Best loss: {current_best_val_loss}')
            return True
        return False

    @classmethod
    def _write_submission_file(cls,
                               csvfile,
                               dataloader_test: DataLoader,
                               model_pred: nn.Module) -> int:
        # Define CSV header columns (x_1, x_3, ..., x_69)
        x_cols = [f"x_{i}" for i in range(1, 70, 2)]
        fieldnames = ["oid_ypos"] + x_cols
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        pbar_test = tqdm(dataloader_test, desc="Generating Submission", unit="batch")
        rows_written = 0
        with torch.no_grad():
            for inputs, original_ids in pbar_test:
                rows_written = cls._write_for_batch(inputs, model_pred, x_cols, writer, original_ids, rows_written)
        return rows_written

    @classmethod
    def _write_for_batch(cls,
                         inputs,
                         model_pred: nn.Module,
                         x_cols: List[str],
                         writer: csv.DictWriter,
                         original_ids: List[str],
                         rows_written: int) -> int:
        # Handle batch size = 1 where original_ids might be a string
        if isinstance(original_ids, str):
            original_ids = [original_ids]
        try:
            inputs = inputs.to(Config.get_device()).float()
            with torch.amp.autocast(
                device_type=Config.get_device().type,
                dtype=Config.get_autocast_dtype(),
                enabled=Config.get_use_cuda(),
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
            cls._logger.exception(
                f"Prediction failed for batch (OID: {original_ids[0] if original_ids else '?'}) : {e}"
            )
        return rows_written

    @classmethod
    def _train_for(cls,
                   epoch: int,
                   model: nn.Module,
                   dataloader_train: DataLoader,
                   dataloader_validation: DataLoader,
                   optimizer: Optimizer,
                   loss_criterion,
                   scheduler: LRScheduler,
                   history: List[Dict[str, Any]],
                   best_val_loss: float):
        cls._logger.info(f"=== Epoch {epoch}/{Config.get_n_epochs()} ===")
        # --- Training Phase ---
        gc.collect()
        if Config.get_use_cuda():
            torch.cuda.empty_cache()
        model_ema = ModelEMA(model, decay=Config.get_weight_decay(), device=Config.get_gpu_local_rank())
        model.train()
        train_losses = []
        progess_bar_train = tqdm(dataloader_train, desc=f"Train E{epoch}", leave=False, unit="batch")
        for i, batch in enumerate(progess_bar_train):
            cls._train_batch(batch, i, optimizer, model, loss_criterion, train_losses, progess_bar_train, model_ema)

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        cls._logger.info(f"Epoch {epoch} Avg Train Loss: {avg_train_loss:.5f}")

        # --- Validation Phase ---
        if dataloader_validation is None:
            cls._logger.warning("Skipping validation phase - no validation DataLoader.")
            history.append({"epoch": epoch, "train_loss": avg_train_loss, "valid_loss": None})
            return best_val_loss # Skip to next epoch

        model.eval()
        val_losses = []
        pbar_val = tqdm(dataloader_validation, desc=f"Valid E{epoch}", leave=False, unit="batch")
        with torch.no_grad():
            for i, batch in enumerate(pbar_val):
                cls._run_validation_batch(batch, model, loss_criterion, val_losses, i, epoch, model_ema)

        avg_val_loss = GPUHelper.gather_and_get_avg_loss_on_same_device(val_losses)
        cls._logger.info(f"Epoch {epoch} Avg Valid Loss: {avg_val_loss:.5f}")
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "valid_loss": avg_val_loss})

        scheduler.step(avg_val_loss)

        cls._log_gpu_stats()

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            cls._save_model(best_val_loss, avg_val_loss, epoch, model, model_ema)

        return best_val_loss

    @classmethod
    def _log_gpu_stats(cls):
        gpu_stats = GPUHelper.get_gpu_memory_statistics()
        cls._logger.info(f"GPU Usage: {gpu_stats[Constants.USAGE_IN_GB]:.2f} GB. -> {gpu_stats[Constants.USAGE_IN_PERCENT]:.2f} %")

    @classmethod
    def _save_model(cls,
                    best_val_loss: float,
                    current_val_loss: float,
                    epoch: int,
                    model: nn.Module,
                    model_ema: ModelEMA):
        cls._remove_old_best_model()
        # Save the new best model
        fname = f"{Config.get_model_prefix()}_epoch_{epoch}_loss_{best_val_loss:.4f}{Constants.EXTN_MODEL}"
        fpath = os.path.join(Config.get_working_dir(), fname)
        cls._logger.info(f"*** New best validation loss: {best_val_loss:.5f}. \n "
                         f"Current Validation Loss: {current_val_loss}. \n "
                         f"Saving model: {fname} ***")
        if model_ema:
            model_state = model_ema.get_module().state_dict()
        else:
            model_state = model.state_dict()
        torch.save(model_state, fpath)

    @classmethod
    def _remove_old_best_model(cls):
        # Clean previous best models before saving new one
        del_pattern = os.path.join(
            Config.get_working_dir(), f"{Config.get_model_prefix()}_epoch_*_loss_*{Constants.EXTN_MODEL}"
        )
        for old_model_path in glob.glob(del_pattern):
            try:
                cls._logger.info(f"Removing model: {os.path.basename(old_model_path)}")
                os.remove(old_model_path)
            except OSError as e:
                cls._logger.exception(f"Could not delete old model {old_model_path}: {e}")

    @classmethod
    def _run_validation_batch(cls,
                              batch: dict,
                              model: nn.Module,
                              loss_criterion,
                              val_losses: List[float],
                              batch_index: int,
                              epoch: int,
                              model_ema: ModelEMA):
        if cls._is_batch_valid(batch) is False:
            return
        try:
            inputs = batch["seis"].to(Config.get_gpu_local_rank(), non_blocking=True).float()
            targets = batch["vel"].to(Config.get_gpu_local_rank(), non_blocking=True).float()
            with torch.amp.autocast(
                device_type=Config.get_device().type,
                dtype=Config.get_autocast_dtype(),
                enabled=Config.get_use_cuda(),
            ):
                if model_ema:
                    outputs = model_ema.get_module()(inputs)
                else:
                    outputs = model(inputs)
                loss = loss_criterion(outputs, targets)
            val_losses.append(loss.item())

            # Plotting validation examples periodically
            if batch_index == 0 and epoch % Config.get_plot_every_n_epochs() == 0:
                # Add validation plotting code here if desired
                pass # Placeholder

        except Exception as e:
            cls._logger.exception(f"Validation batch {batch_index} failed: {e}")
            if isinstance(e, torch.cuda.OutOfMemoryError):
                cls._logger.error("CUDA Out of Memory during validation. Exiting.")
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
                     pbar_train: tqdm,
                     model_ema: ModelEMA):
        if cls._is_batch_valid(batch) is False:
            return
        try:
            inputs = batch[Constants.SEIS].to(Config.get_gpu_local_rank(), non_blocking=True).float()
            targets = batch[Constants.VEL].to(Config.get_gpu_local_rank(), non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            # Use Automatic Mixed Precision (AMP) if on CUDA
            with torch.amp.autocast(
                device_type=Config.get_device().type,
                dtype=Config.get_autocast_dtype(),
                enabled=Config.get_use_cuda(),
            ):
                outputs = model(inputs)
                loss = loss_criterion(outputs, targets)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if model_ema is not None:
                model_ema.update(model)

            # Update progress bar description
            if batch_index % 100 == 0:
                memory_statistics = GPUHelper.get_gpu_memory_statistics()
                pbar_train.set_postfix(loss=f"{np.mean(train_losses):.5f}",
                                       gpu_usage=f'{memory_statistics[Constants.USAGE_IN_PERCENT]:.3f}')

        except Exception as e:
            cls._logger.exception(f"Training batch {batch_index} failed: {e}")
            # Stop training on OOM error
            if isinstance(e, torch.cuda.OutOfMemoryError):
                cls._logger.error("CUDA Out of Memory during training. Exiting.")
                raise e
            # Continue on other errors if desired, or raise
            # raise e # Uncomment to stop on any training error

    @staticmethod
    def _is_batch_valid(batch: dict):
        if not batch or Constants.SEIS not in batch or Constants.VEL not in batch:
            return False
        return True
