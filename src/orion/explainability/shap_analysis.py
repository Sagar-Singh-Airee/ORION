"""Dependency-free feature permutation attribution."""
from __future__ import annotations

import numpy as np


def permutation_importance(predict, features: np.ndarray, targets: np.ndarray, score, repeats: int = 5, seed: int = 42) -> np.ndarray:
    baseline = score(targets, predict(features)); rng = np.random.default_rng(seed); importances = np.zeros(features.shape[1])
    for column in range(features.shape[1]):
        values = []
        for _ in range(repeats):
            shuffled = features.copy(); rng.shuffle(shuffled[:, column]); values.append(baseline - score(targets, predict(shuffled)))
        importances[column] = np.mean(values)
    return importances
