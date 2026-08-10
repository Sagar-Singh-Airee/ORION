"""Collation for fixed- or variable-length multi-instance MRI batches."""
from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

__all__ = ["variable_length_collate"]


def _validate_images(images: list[torch.Tensor]) -> None:
    """Fail with a specific, indexed message rather than a generic torch shape/dtype error."""
    bad_shapes = [(index, tuple(image.shape)) for index, image in enumerate(images) if image.ndim != 4]
    if bad_shapes:
        raise ValueError(f"Each image must have shape (S, C, H, W); got bad shape(s) at indices: {bad_shapes}")

    zero_slice_indices = [index for index, image in enumerate(images) if image.shape[0] == 0]
    if zero_slice_indices:
        raise ValueError(f"Image(s) at batch index {zero_slice_indices} have 0 slices")

    reference_dtype, reference_device = images[0].dtype, images[0].device
    reference_shape = images[0].shape[1:]
    for index, image in enumerate(images):
        if image.dtype != reference_dtype:
            raise ValueError(
                f"All images in a batch must share a dtype; index 0 is {reference_dtype}, index {index} is {image.dtype}"
            )
        if image.device != reference_device:
            raise ValueError(
                f"All images in a batch must share a device; index 0 is {reference_device}, index {index} is {image.device}"
            )
        if image.shape[1:] != reference_shape:
            raise ValueError(
                f"All images in a batch must share (C, H, W); index 0 has {tuple(reference_shape)}, "
                f"index {index} has {tuple(image.shape[1:])}"
            )


def _build_slice_mask(batch: list[dict[str, Any]], lengths: torch.Tensor, max_slices: int) -> torch.Tensor:
    """Default mask marks the first `length` slices of each item valid. An explicit
    'slice_mask' in an item overrides that (e.g. to exclude a corrupted mid-sequence
    slice), and must cover exactly that item's own slice count — a mismatch here would
    otherwise silently misalign which slices the model attends to.
    """
    mask = torch.arange(max_slices, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    for index, item in enumerate(batch):
        supplied_mask = item.get("slice_mask")
        if supplied_mask is None:
            continue
        supplied = torch.as_tensor(supplied_mask, dtype=torch.bool, device=lengths.device)
        if supplied.numel() != int(lengths[index]):
            raise ValueError(
                f"slice_mask length ({supplied.numel()}) does not match image slice count "
                f"({int(lengths[index])}) at batch index {index}"
            )
        mask[index, : lengths[index]] = supplied
    return mask


def _collate_extra_keys(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate every key besides 'image'/'slice_mask': stack if all-tensor, else keep as a list."""
    reference_keys = set(batch[0].keys())
    for index, item in enumerate(batch[1:], start=1):
        if set(item.keys()) != reference_keys:
            raise ValueError(
                f"All batch items must have the same keys; index 0 has {sorted(reference_keys)}, "
                f"index {index} has {sorted(item.keys())}"
            )

    collated: dict[str, Any] = {}
    for key in reference_keys:
        if key in {"image", "slice_mask"}:
            continue
        values = [item[key] for item in batch]
        if all(isinstance(value, torch.Tensor) for value in values):
            try:
                collated[key] = torch.stack(values)
            except RuntimeError as exc:
                shapes = [tuple(value.shape) for value in values]
                raise ValueError(f"Cannot stack tensors for key {key!r}; got shapes {shapes}") from exc
        else:
            collated[key] = values
    return collated


def variable_length_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of per-study samples into a padded, mask-aware batch.

    Each item must contain 'image': a (S, C, H, W) tensor, where S (slice count) may
    vary across items but C/H/W, dtype, and device must match across the batch. An
    optional 'slice_mask': (S,) bool/int array marks which of that item's own slices
    are valid; if omitted, all S slices are treated as valid.

    Returns:
        A dict with:
          - 'image': (B, max_S, C, H, W), zero-padded past each item's own length.
          - 'slice_mask': (B, max_S) bool, True for real slices, False for padding.
          - every other key present in every item: stacked into a tensor if all
            values are tensors, else returned as a plain list (e.g. string ids,
            metadata dicts).
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    images = [item["image"] for item in batch]
    _validate_images(images)

    lengths = torch.tensor([image.shape[0] for image in images], dtype=torch.long, device=images[0].device)
    max_slices = int(lengths.max())

    collated: dict[str, Any] = {
        "image": pad_sequence(images, batch_first=True),
        "slice_mask": _build_slice_mask(batch, lengths, max_slices),
    }
    collated.update(_collate_extra_keys(batch))
    return collated