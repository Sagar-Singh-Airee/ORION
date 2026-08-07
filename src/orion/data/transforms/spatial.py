"""Spatial transforms that preserve the MRI slice-axis contract."""
from __future__ import annotations

import numpy as np


def horizontal_flip(volume: np.ndarray) -> np.ndarray:
    if volume.ndim < 2:
        raise ValueError("volume must have at least two spatial dimensions")
    return np.flip(volume, axis=-1).copy()


def vertical_flip(volume: np.ndarray) -> np.ndarray:
    if volume.ndim < 2:
        raise ValueError("volume must have at least two spatial dimensions")
    return np.flip(volume, axis=-2).copy()
