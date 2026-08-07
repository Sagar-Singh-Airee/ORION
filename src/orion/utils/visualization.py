"""Compact plotting helpers for experiment notebooks."""
from __future__ import annotations

import numpy as np


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32); low, high = np.percentile(image, [1, 99])
    return np.clip((image - low) / max(high - low, 1e-6), 0, 1)
