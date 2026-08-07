from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import LOSSES


@LOSSES.register("nt_xent")
class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0: raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape or left.ndim != 2: raise ValueError("Embeddings must have the same [B, C] shape")
        logits = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).T / self.temperature
        targets = torch.arange(len(left), device=left.device)
        return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
