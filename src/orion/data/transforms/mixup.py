"""Batch-level mixup for soft multilabel targets."""
from __future__ import annotations

import torch


def mixup_batch(images: torch.Tensor, targets: torch.Tensor, alpha: float = 0.4) -> tuple[torch.Tensor, torch.Tensor, float]:
    if images.size(0) != targets.size(0):
        raise ValueError("images and targets must have the same batch size")
    if alpha <= 0 or images.size(0) < 2:
        return images, targets, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    index = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1.0 - lam) * images[index], lam * targets + (1.0 - lam) * targets[index], lam
