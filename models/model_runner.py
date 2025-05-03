import gc
import glob
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from configs.config import Config
from helpers.constants import Constants

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
