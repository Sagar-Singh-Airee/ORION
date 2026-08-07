"""ONNX export with dynamic batch and slice dimensions."""
from __future__ import annotations

from pathlib import Path

import torch


def export_onnx(model: torch.nn.Module, output_path: str | Path, image_size: tuple[int, int] = (384, 384), num_slices: int = 24, num_classes: int = 12) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    class _ImageOnlyWrapper(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module):
            super().__init__(); self.wrapped = wrapped
        def forward(self, image: torch.Tensor, slice_mask: torch.Tensor) -> torch.Tensor:
            return self.wrapped(image=image, slice_mask=slice_mask)
    wrapper = _ImageOnlyWrapper(model)
    image = torch.zeros(1, num_slices, 1, *image_size)
    mask = torch.ones(1, num_slices, dtype=torch.bool)
    torch.onnx.export(wrapper, (image, mask), path, input_names=["image", "slice_mask"], output_names=["logits"], dynamic_axes={"image": {0: "batch", 1: "slices"}, "slice_mask": {0: "batch", 1: "slices"}, "logits": {0: "batch"}}, opset_version=17)
    return path
