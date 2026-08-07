"""Pruning helpers; callers should fine-tune after pruning."""
from __future__ import annotations

import torch
import torch.nn.utils.prune as prune


def global_unstructured_prune(model: torch.nn.Module, amount: float = 0.2) -> None:
    if not 0 <= amount < 1: raise ValueError("amount must be in [0, 1)")
    parameters = [(module, "weight") for module in model.modules() if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]
    prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=amount)


def remove_pruning_reparameterization(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "weight_orig"): prune.remove(module, "weight")
