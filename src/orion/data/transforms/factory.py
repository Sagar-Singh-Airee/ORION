"""Configurable, medically conservative 2-D augmentation factories."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:  # pragma: no cover
    A = None  # type: ignore[assignment]
    ALBUMENTATIONS_AVAILABLE = False


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _section(config: Any) -> Any:
    data = _get(config, "data", config)
    return _get(data, "augmentation", data)


def _image_size(config: Any) -> tuple[int, int]:
    data = _get(config, "data", config)
    preprocessing = _get(data, "preprocessing", data)
    output = _get(preprocessing, "output", preprocessing)
    size = _get(output, "image_size", _get(data, "image_size", (384, 384)))
    return int(size[0]), int(size[1])


def _enabled(section: Any, name: str) -> Any | None:
    value = _get(section, name, {})
    return value if _get(value, "enabled", False) else None


def _identity(image: np.ndarray | None = None, **_: Any) -> dict[str, np.ndarray | None]:
    return {"image": image}


def create_train_transforms(config: Any) -> Callable[..., dict[str, Any]]:
    """Build an Albumentations pipeline; no silent tensor conversion occurs here."""
    if not ALBUMENTATIONS_AVAILABLE:
        return _identity
    aug = _section(config)
    height, width = _image_size(config)
    spatial = _get(aug, "spatial", {})
    intensity = _get(aug, "intensity", {})
    transforms: list[Any] = []
    if option := _enabled(spatial, "hflip"):
        transforms.append(A.HorizontalFlip(p=float(_get(option, "p", 0.5))))
    if option := _enabled(spatial, "vflip"):
        transforms.append(A.VerticalFlip(p=float(_get(option, "p", 0.5))))
    if option := _enabled(spatial, "rotation"):
        transforms.append(A.Rotate(limit=float(_get(option, "limit", 10)), p=float(_get(option, "p", 0.5))))
    if option := _enabled(spatial, "scale"):
        transforms.append(A.RandomScale(scale_limit=float(_get(option, "scale_limit", 0.1)), p=float(_get(option, "p", 0.3))))
    crop = _enabled(spatial, "random_crop")
    if crop:
        transforms.append(
            A.RandomResizedCrop(
                height=height,
                width=width,
                scale=tuple(_get(crop, "scale", (0.9, 1.0))),
                ratio=tuple(_get(crop, "ratio", (0.95, 1.05))),
                p=float(_get(crop, "p", 1.0)),
            )
        )
    else:
        transforms.append(A.Resize(height=height, width=width))
    if option := _enabled(intensity, "brightness_contrast"):
        transforms.append(A.RandomBrightnessContrast(brightness_limit=float(_get(option, "brightness_limit", 0.15)), contrast_limit=float(_get(option, "contrast_limit", 0.15)), p=float(_get(option, "p", 0.5))))
    if option := _enabled(intensity, "gaussian_noise"):
        transforms.append(A.GaussNoise(var_limit=tuple(_get(option, "var_limit", (5.0, 20.0))), p=float(_get(option, "p", 0.3))))
    if option := _enabled(intensity, "gaussian_blur"):
        transforms.append(A.GaussianBlur(blur_limit=tuple(_get(option, "blur_limit", (3, 5))), p=float(_get(option, "p", 0.2))))
    if option := _enabled(intensity, "gamma"):
        transforms.append(A.RandomGamma(gamma_limit=tuple(_get(option, "gamma_limit", (80, 120))), p=float(_get(option, "p", 0.2))))
    # Replay keeps spatial/intensity choices coherent across the slices in one MRI.
    return A.ReplayCompose(transforms)


def create_val_transforms(config: Any) -> Callable[..., dict[str, Any]]:
    if not ALBUMENTATIONS_AVAILABLE:
        return _identity
    height, width = _image_size(config)
    return A.Compose([A.Resize(height=height, width=width)])
