"""Memory telemetry that gracefully works without CUDA."""
from __future__ import annotations

import torch


def cuda_memory_mb(device: torch.device | None = None) -> dict[str, float]:
    if not torch.cuda.is_available(): return {"allocated": 0.0, "reserved": 0.0}
    return {"allocated": torch.cuda.memory_allocated(device) / 2**20, "reserved": torch.cuda.memory_reserved(device) / 2**20}
