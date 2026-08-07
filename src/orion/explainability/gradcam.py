"""Grad-CAM without OpenCV or in-place activation mutation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model, self.target_layer = model, target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [target_layer.register_forward_hook(self._save_activation), target_layer.register_full_backward_hook(self._save_gradient)]

    def _save_activation(self, _: nn.Module, __: tuple[object, ...], output: torch.Tensor) -> None:
        self.activations = output

    def _save_gradient(self, _: nn.Module, __: tuple[object, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_output[0]

    def close(self) -> None:
        for handle in self._handles: handle.remove()
        self._handles.clear()

    def __call__(self, image: torch.Tensor, target_class: int, **forward_kwargs: object) -> torch.Tensor:
        self.model.eval(); self.model.zero_grad(set_to_none=True)
        logits = self.model(image, **forward_kwargs)
        if self.activations is None: raise RuntimeError("Target layer did not produce activations")
        logits[:, target_class].sum().backward()
        if self.gradients is None: raise RuntimeError("Target layer did not receive gradients")
        if self.activations.ndim != 4: raise ValueError("Grad-CAM target layer must output [B, C, H, W]")
        weights = self.gradients.mean(dim=(-2, -1), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        maximum = cam.amax(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(cam.dtype).eps)
        return (cam / maximum).squeeze(1)
