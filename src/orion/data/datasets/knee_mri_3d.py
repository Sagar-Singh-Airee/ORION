"""3-D CNN view of the canonical study dataset."""
from __future__ import annotations

from typing import Any

from .knee_mri import KneeMRIDataset


class KneeMRI3DDataset(KneeMRIDataset):
    """Return images as `(C, S, H, W)` rather than MIL's `(S, C, H, W)`."""
    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        item["image"] = item["image"].permute(1, 0, 2, 3).contiguous()
        return item
