"""Safe test-time augmentations for MRI tensor batches."""
from __future__ import annotations

from collections.abc import Callable, Iterable

import torch


def identity(images: torch.Tensor) -> torch.Tensor:
    return images


def hflip(images: torch.Tensor) -> torch.Tensor:
    return torch.flip(images, dims=(-1,))


def vflip(images: torch.Tensor) -> torch.Tensor:
    return torch.flip(images, dims=(-2,))


def rotate_90(images: torch.Tensor) -> torch.Tensor:
    return torch.rot90(images, k=1, dims=(-2, -1))


TTA_TRANSFORMS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": identity, "original": identity, "hflip": hflip, "vflip": vflip, "rotate_90": rotate_90,
}


def resolve_tta(names: Iterable[str]) -> list[Callable[[torch.Tensor], torch.Tensor]]:
    resolved = []
    for name in names:
        if name not in TTA_TRANSFORMS:
            raise ValueError(f"Unknown TTA {name!r}; choices: {sorted(TTA_TRANSFORMS)}")
        resolved.append(TTA_TRANSFORMS[name])
    return resolved or [identity]
