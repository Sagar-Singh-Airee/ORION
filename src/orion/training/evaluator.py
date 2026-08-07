"""Validation collection used by the trainer and cross-validation scripts."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from ..evaluation.metrics import calculate_metrics


@torch.inference_mode()
def evaluate_model(model: torch.nn.Module, loader: Iterable[dict[str, Any]], device: torch.device, label_names: list[str]) -> dict[str, float]:
    model.eval(); targets, probabilities = [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch.get("slice_mask")
        if mask is not None: mask = mask.to(device, non_blocking=True)
        text = None
        if "input_ids" in batch:
            text = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch.get("attention_mask", None)}
            if text["attention_mask"] is not None: text["attention_mask"] = text["attention_mask"].to(device)
        output = model(image=image, text_inputs=text, slice_mask=mask)
        targets.append(batch["label"].cpu().numpy())
        probabilities.append(torch.sigmoid(output).cpu().numpy())
    if not targets: raise ValueError("Validation loader yielded no batches")
    return calculate_metrics(np.concatenate(targets), np.concatenate(probabilities), label_names)
