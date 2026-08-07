"""Collation for fixed- or variable-length multi-instance MRI batches."""
from __future__ import annotations

from typing import Any

import torch


def variable_length_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    images = [item["image"] for item in batch]
    if any(image.ndim != 4 for image in images):
        raise ValueError("Each image must have shape (S, C, H, W)")
    max_slices = max(image.shape[0] for image in images)
    padded_images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for item, image in zip(batch, images, strict=True):
        length = image.shape[0]
        if length < max_slices:
            padded = image.new_zeros((max_slices, *image.shape[1:]))
            padded[:length] = image
            image = padded
        padded_images.append(image)
        supplied_mask = item.get("slice_mask")
        if supplied_mask is None:
            mask = torch.zeros(max_slices, dtype=torch.bool)
            mask[:length] = True
        else:
            mask = torch.as_tensor(supplied_mask, dtype=torch.bool)
            if len(mask) < max_slices:
                mask = torch.cat((mask, torch.zeros(max_slices - len(mask), dtype=torch.bool)))
        masks.append(mask)
    collated: dict[str, Any] = {"image": torch.stack(padded_images), "slice_mask": torch.stack(masks)}
    for key in batch[0]:
        if key in {"image", "slice_mask"}:
            continue
        values = [item.get(key) for item in batch]
        collated[key] = torch.stack(values) if all(isinstance(value, torch.Tensor) for value in values) else values
    return collated
