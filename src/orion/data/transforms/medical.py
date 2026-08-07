"""MRI-specific, physics-inspired but lightweight volume perturbations."""
from __future__ import annotations

import numpy as np


def multiplicative_bias_field(volume: np.ndarray, strength: float = 0.2) -> np.ndarray:
    if strength < 0:
        raise ValueError("strength must be non-negative")
    _, height, width = volume.shape
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    field = 1 + strength * (xx * xx + yy * yy - 2 / 3)
    return (volume.astype(np.float32) * field[None]).astype(np.float32)
