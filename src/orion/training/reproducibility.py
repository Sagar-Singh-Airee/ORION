"""Seed everything. On Kaggle/Colab shared GPUs, full `cudnn.deterministic=True` can cost
10-30% throughput on conv-heavy backbones — exposed as a flag so speed-priority runs
(early iteration) can trade determinism for speed, while final/submission runs lock it down.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Pass to DataLoader(worker_init_fn=...) so each worker's numpy/random streams
    diverge deterministically from the base seed instead of all workers sharing state."""
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def seeded_generator(seed: int) -> torch.Generator:
    """For DataLoader(generator=...) so shuffling order is reproducible per fold/run."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g