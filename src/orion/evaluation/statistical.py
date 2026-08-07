"""Confidence intervals for model-selection reports."""
from __future__ import annotations

import numpy as np

from .metrics import macro_auc


def bootstrap_macro_auc(
    targets: np.ndarray,
    probabilities: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return estimate and percentile CI, resampling studies rather than labels."""
    if not 0 < confidence < 1 or n_bootstrap <= 0:
        raise ValueError("confidence must be in (0, 1) and n_bootstrap positive")
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    estimate = macro_auc(targets, probabilities)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(targets), len(targets))
        value = macro_auc(targets[index], probabilities[index])
        if np.isfinite(value):
            samples.append(value)
    alpha = (1 - confidence) / 2
    return estimate, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))
