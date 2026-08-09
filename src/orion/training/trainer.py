"""Small, explicit trainer with mixed precision, masking, and OOF validation."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..evaluation.metrics import calculate_metrics
from ..models.ema import ModelEMA
from ..utils.io import save_checkpoint
from ..utils.wandb_utils import log_metrics
from .mixed_precision import AMPManager


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)


class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device, train_loader: Iterable[dict[str, Any]], val_loader: Iterable[dict[str, Any]], config: Any, scheduler: Any = None, ema: ModelEMA | None = None, label_names: list[str] | None = None):
        self.model, self.optimizer, self.criterion, self.device = model.to(device), optimizer, criterion.to(device), device
        self.train_loader, self.val_loader, self.scheduler, self.ema, self.config = train_loader, val_loader, scheduler, ema, config
        training = _get(config, "training", {})
        self.epochs = int(_get(training, "max_epochs", _get(training, "epochs", 1)))
        self.grad_accum_steps = int(_get(training, "gradient_accumulation_steps", _get(training, "grad_accum_steps", 1)))
        self.clip_val = float(_get(training, "gradient_clip_val", _get(training, "grad_clip_norm", 0.0)))
        self.patience = int(_get(training, "early_stopping_patience", 0))
        self.amp = AMPManager(bool(_get(training, "mixed_precision", _get(training, "amp", True))), device.type)
        self.label_names = label_names or [f"class_{index}" for index in range(int(_get(_get(config, "labels", {}), "num_classes", 12)))]
        self.global_step = 0

    def _set_backbone_trainable(self, epoch: int) -> None:
        """Freeze only backbone parameters for configured warm-up epochs."""
        backbone = getattr(self.model, "vision_backbone", None)
        freeze_until = int(getattr(self.model, "freeze_backbone_until_epoch", 0))
        if backbone is None:
            return
        trainable = epoch >= freeze_until
        for parameter in backbone.parameters():
            parameter.requires_grad_(trainable)
        # Frozen BatchNorm/dropout must not keep changing the supposedly frozen encoder.
        backbone.train(trainable)

    def _optimizer_step(self, gradient_correction: float = 1.0) -> None:
        """Step once, correcting a final short accumulation window if needed."""
        if gradient_correction != 1.0:
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(gradient_correction)
        if self.clip_val > 0:
            self.amp.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_val)
        self.amp.step(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema:
            self.ema.update(self.model)
        self.global_step += 1

    def _move_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor] | None]:
        image, targets = batch["image"].to(self.device, non_blocking=True), batch["label"].to(self.device, non_blocking=True)
        mask = batch.get("slice_mask"); mask = mask.to(self.device, non_blocking=True) if mask is not None else None
        text = None
        if "input_ids" in batch:
            text = {"input_ids": batch["input_ids"].to(self.device, non_blocking=True)}
            if batch.get("attention_mask") is not None: text["attention_mask"] = batch["attention_mask"].to(self.device, non_blocking=True)
        return image, targets, mask, text

    def train_epoch(self, epoch: int) -> float:
        self.model.train(); self._set_backbone_trainable(epoch); self.optimizer.zero_grad(set_to_none=True); total = 0.0; batches = 0
        for batch_index, batch in enumerate(self.train_loader):
            images, targets, mask, text = self._move_batch(batch)
            with self.amp.autocast():
                loss = self.criterion(self.model(image=images, text_inputs=text, slice_mask=mask), targets)
                scaled_loss = loss / self.grad_accum_steps
            self.amp.backward(scaled_loss); batches += 1; total += float(loss.detach())
            if (batch_index + 1) % self.grad_accum_steps == 0:
                self._optimizer_step()
        # Flush a final partial accumulation window.
        if batches and batches % self.grad_accum_steps:
            self._optimizer_step(self.grad_accum_steps / (batches % self.grad_accum_steps))
        return total / max(1, batches)

    @torch.inference_mode()
    def validate(self) -> dict[str, float]:
        model = self.ema.ema_model if self.ema else self.model; model.eval(); targets, probabilities = [], []
        for batch in self.val_loader:
            images, labels, mask, text = self._move_batch(batch)
            targets.append(labels.cpu().numpy()); probabilities.append(torch.sigmoid(model(image=images, text_inputs=text, slice_mask=mask)).cpu().numpy())
        if not targets:
            raise ValueError("Validation loader yielded no batches")
        return calculate_metrics(np.concatenate(targets), np.concatenate(probabilities), self.label_names)

    def fit(self, output_dir: str | Path | None = None) -> list[dict[str, float]]:
        output_dir = Path(output_dir) if output_dir else None; history=[]; best=float("-inf"); waits=0
        for epoch in range(self.epochs):
            record = {"epoch": float(epoch), "train_loss": self.train_epoch(epoch), **{f"val_{key}": value for key, value in self.validate().items()}}
            if self.scheduler: self.scheduler.step()
            score = record.get("val_macro_auc_roc", float("-inf"))
            if not np.isfinite(score):
                raise RuntimeError("Validation macro AUC is undefined; each fold needs at least one evaluable label")
            improved = score > best
            if improved: best, waits = score, 0
            else: waits += 1
            if output_dir:
                evaluation_model = self.ema.ema_model if self.ema else self.model
                save_checkpoint(
                    {"model": evaluation_model.state_dict(), "raw_model": self.model.state_dict(), "ema": self.ema.state_dict() if self.ema else None, "optimizer": self.optimizer.state_dict(), "epoch": epoch, "metrics": record},
                    improved,
                    output_dir,
                    "last.pth",
                )
            log_metrics(record, self.global_step); history.append(record)
            if self.patience and waits >= self.patience: break
        return history
