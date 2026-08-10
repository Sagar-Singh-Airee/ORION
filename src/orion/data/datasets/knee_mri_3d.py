"""3-D CNN view of the canonical study dataset."""
from __future__ import annotations

from typing import Any

from .knee_mri import KneeMRIDataset

__all__ = ["KneeMRI3DDataset"]


class KneeMRI3DDataset(KneeMRIDataset):
    """Return images as `(C, S, H, W)` — the per-sample layout `nn.Conv3d` expects
    (slices double as the volumetric "depth" dimension) — rather than the MIL
    architectures' `(S, C, H, W)`.

    Important: `variable_length_collate` (orion.data.datasets.collate) pads and masks
    along dim 0 of each item's image — the *slice* dimension in MIL's `(S, C, H, W)`
    layout, but the *channel* dimension here. Do not wire this dataset to
    `variable_length_collate`; it's only correct where every study has already been
    resampled to a fixed slice count (e.g. via `cfg.data.num_slices`) and can use the
    standard batch collate instead.

    This permutation runs after the inherited transform pipeline — `KneeMRIDataset.
    __getitem__` already returns a transformed sample in `(S, C, H, W)` — so
    augmentations operate on the standard layout before this final axis reorder.
    """

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        image = item["image"]
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (S, C, H, W) from the base dataset, got {tuple(image.shape)}")
        item["image"] = image.permute(1, 0, 2, 3).contiguous()
        return item