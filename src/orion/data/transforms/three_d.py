"""Volume-level MRI augmentations with deterministic probability handling."""
from __future__ import annotations

import numpy as np


def random_slice_dropout(volume: np.ndarray, probability: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    if not 0 <= probability < 1:
        raise ValueError("probability must be in [0, 1)")
    rng = rng or np.random.default_rng()
    output = volume.copy()
    drop = rng.random(len(output)) < probability
    if drop.all() and len(drop):
        drop[rng.integers(len(drop))] = False
    output[drop] = 0
    return output
