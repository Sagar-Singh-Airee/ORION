"""Utilities for selecting high-confidence pseudo/weak labels."""
from __future__ import annotations

import numpy as np


def confident_labels(probabilities: np.ndarray, positive_threshold: float = 0.9, negative_threshold: float = 0.1) -> np.ndarray:
    if not 0 <= negative_threshold < positive_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= negative < positive <= 1")
    probabilities = np.asarray(probabilities)
    labels = np.full(probabilities.shape, -1, dtype=np.int8)
    labels[probabilities >= positive_threshold] = 1
    labels[probabilities <= negative_threshold] = 0
    return labels
