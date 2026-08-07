"""NumPy MRI intensity transforms useful outside Albumentations."""
from __future__ import annotations

import numpy as np


def add_gaussian_noise(volume: np.ndarray, std: float, rng: np.random.Generator | None = None) -> np.ndarray:
    if std < 0:
        raise ValueError("std must be non-negative")
    rng = rng or np.random.default_rng()
    return (volume.astype(np.float32) + rng.normal(0, std, volume.shape)).astype(np.float32)


def gamma_correct(volume: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return np.power(np.clip(volume, 0.0, 1.0), gamma).astype(np.float32)
