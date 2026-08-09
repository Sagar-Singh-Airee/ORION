"""Batched prediction with optional TTA and no training-time side effects."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from .tta import resolve_tta


class Predictor:
    def __init__(self, model: torch.nn.Module, device: torch.device | str | None = None, tta: Iterable[str] = ("identity",)):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.tta = resolve_tta(tta)
        self._use_amp = self.device.type == "cuda"

    @torch.inference_mode()
    def predict_loader(self, loader: Iterable[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
        ids: list[str] = []
        predictions: list[np.ndarray] = []
        for batch in loader:
            image = batch["image"].to(self.device, non_blocking=True)
            slice_mask = batch.get("slice_mask")
            if slice_mask is not None:
                slice_mask = slice_mask.to(self.device, non_blocking=True)
            text_inputs = None
            if "input_ids" in batch:
                text_inputs = {"input_ids": batch["input_ids"].to(self.device, non_blocking=True)}
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    text_inputs["attention_mask"] = attention_mask.to(self.device, non_blocking=True)
            summed_logits = None
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self._use_amp):
                for transform in self.tta:
                    output = self.model(image=transform(image), text_inputs=text_inputs, slice_mask=slice_mask)
                    summed_logits = output if summed_logits is None else summed_logits + output
            probabilities = torch.sigmoid(summed_logits / len(self.tta)).float().cpu().numpy()
            predictions.append(probabilities)
            ids.extend(str(value) for value in batch["study_uid"])
        if not predictions:
            return ids, np.empty((0, 0), dtype=np.float32)
        return ids, np.concatenate(predictions).astype(np.float32)
