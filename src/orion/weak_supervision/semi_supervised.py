"""Confidence masking used by teacher-student training loops."""
from __future__ import annotations

import numpy as np


def pseudo_label_mask(probabilities: np.ndarray, threshold: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities)
    confidence = np.maximum(probabilities, 1 - probabilities)
    return (probabilities >= 0.5).astype(np.float32), confidence >= threshold
