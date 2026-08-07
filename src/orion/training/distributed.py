"""Minimal DDP lifecycle; a no-op when launched normally on one GPU."""
from __future__ import annotations
import os
import torch
import torch.distributed as dist

def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ: return 0, 0, 1
    rank, local_rank, world_size = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank); dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size

def is_main_process() -> bool: return not dist.is_initialized() or dist.get_rank() == 0

def cleanup_distributed() -> None:
    if dist.is_initialized(): dist.destroy_process_group()
