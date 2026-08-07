"""ONNX export with dynamic batch and slice dimensions."""
from __future__ import annotations

from pathlib import Path

import torch


def export_onnx(model: torch.nn.Module, output_path: str | Path, image_size: tuple[int, int] = (384, 384), num_slices: int = 24, num_classes: int = 12) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    image = torch.zeros(1, num_slices, 1, *image_size)
    mask = torch.ones(1, num_slices, dtype=torch.bool)
    torch.onnx.export(model, (image, None, mask), path, input_names=["image", "text_inputs", "slice_mask"], output_names=["logits"], dynamic_axes={"image": {0: "batch", 1: "slices"}, "slice_mask": {0: "batch", 1: "slices"}, "logits": {0: "batch"}}, opset_version=17)
    return path
