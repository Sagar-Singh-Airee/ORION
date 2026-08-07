"""AMP wrapper that is safe on CUDA, MPS, and CPU fallback runs."""
from __future__ import annotations

from contextlib import nullcontext

import torch


class AMPManager:
    def __init__(self, enabled: bool = True, device_type: str = "cuda"):
        self.device_type = device_type
        self.enabled = bool(enabled and device_type == "cuda" and torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=True) if self.enabled else nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.step(optimizer); self.scaler.update()
