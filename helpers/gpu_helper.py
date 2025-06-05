import ctypes
import gc
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist

from configs.config import Config
from helpers.constants import Constants

class GPUHelper:

    _ONE_GB = 1024**3

    @staticmethod
    def clean_all_memory(deep=False):
        """More thorough memory cleanup"""
        gc.collect()
        if deep:
            # If possible, gives memory back to the system (via negative arguments to sbrk) if there is unused memory at the `high' end of
            # the malloc pool. You can call this after freeing large blocks of memory to potentially reduce the system-level memory requirements
            # of a program. If the argument is zero, only the minimum amount of memory to maintain internal data structures will be left.
            # Next allocation will need the program to get more memory from the system.
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except:
                pass

    @classmethod
    def get_gpu_memory_statistics(cls) -> Dict[str, float]:
        free, total = cls._get_gpu_free_and_total_memory()
        mem_used = (total - free)
        mem_used_percent = mem_used * 100 / total
        return {Constants.USAGE_IN_GB: mem_used / cls._ONE_GB,
                Constants.USAGE_IN_PERCENT: mem_used_percent,
                Constants.TOTAL_IN_GB: total / cls._ONE_GB}

    @classmethod
    def run_on_multi_gpu_if_required(cls, use_multiple_gpus: bool):
        if not use_multiple_gpus:
            return
        Config.initialize_gpu_config(use_multiple_gpus)
        rank = Config.get_gpu_local_rank()
        world_size = Config.get_gpu_world_size()
        cls._setup_for_multi_gpu(rank, world_size)
        cls._print_info(rank, world_size)

    @classmethod
    def cleanup_multi_gpu_setup(cls):
        if cls._is_single_gpu_setup():
            return
        dist.barrier()
        dist.destroy_process_group()
        Config.initialize_gpu_config(False)

    @classmethod
    def gather_and_get_avg_loss_on_same_device(cls, val_losses: List[float]) -> float:
        if cls._is_single_gpu_setup():
            return np.mean(val_losses) if val_losses else float("inf")
        v = torch.tensor([sum(val_losses), len(val_losses)], device=Config.get_gpu_local_rank())
        torch.distributed.all_reduce(v, op=dist.ReduceOp.SUM)
        return (v[0] / v[1]).item()

    @classmethod
    def _get_gpu_free_and_total_memory(cls) -> Tuple[float, float]:
        free_across_gpus = 0
        total_across_gpus = 0
        for i in range(Config.get_gpu_world_size()):
            free, total = torch.cuda.mem_get_info(device=0)
            free_across_gpus += free
            total_across_gpus += total
        return free_across_gpus, total_across_gpus

    @staticmethod
    def _setup_for_multi_gpu(rank: int, world_size: int):
        torch.cuda.set_device(rank)
        dist.init_process_group(Config.get_multi_gpu_backend(), rank=rank, world_size=world_size)

    @classmethod
    def _print_info(cls, rank: int, world_size: int):
        gpu_info = cls.get_gpu_memory_statistics()
        time.sleep(rank)
        print(f"Rank: {rank}, World size: {world_size}, GPU memory: {gpu_info[Constants.TOTAL_IN_GB]:.2f}GB", flush=True)
        time.sleep(world_size - rank)

    @staticmethod
    def _is_single_gpu_setup() -> bool:
        return not Config.get_use_multiple_gpus()
