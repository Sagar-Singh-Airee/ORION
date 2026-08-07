"""Estimate weak-label agreement with a small expert-labelled audit subset."""
from __future__ import annotations

import numpy as np


def label_function_accuracy(label_matrix: np.ndarray, expert_targets: np.ndarray) -> np.ndarray:
    label_matrix, expert_targets = np.asarray(label_matrix), np.asarray(expert_targets)
    if label_matrix.ndim != 2 or expert_targets.shape != (label_matrix.shape[0],):
        raise ValueError("Expected L=(N, M) and targets=(N,)")
    scores = np.full(label_matrix.shape[1], np.nan)
    for lf in range(label_matrix.shape[1]):
        known = (label_matrix[:, lf] >= 0) & (expert_targets >= 0)
        if known.any(): scores[lf] = (label_matrix[known, lf] == expert_targets[known]).mean()
    return scores
