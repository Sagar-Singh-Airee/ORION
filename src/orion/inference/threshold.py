"""Threshold fitting for clinical review; Kaggle ROC-AUC uses raw probabilities."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_curve


def fit_thresholds(targets: np.ndarray, probabilities: np.ndarray, method: str = "youden") -> np.ndarray:
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    if targets.shape != probabilities.shape or targets.ndim != 2:
        raise ValueError("targets and probabilities must be same (N, C) shape")
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    for index in range(targets.shape[1]):
        known = targets[:, index] >= 0
        truth, probability = targets[known, index], probabilities[known, index]
        if len(truth) == 0 or np.unique(truth).size < 2:
            continue
        if method == "youden":
            fpr, tpr, candidates = roc_curve(truth, probability)
            thresholds[index] = candidates[np.argmax(tpr - fpr)]
        elif method == "f1":
            candidates = np.unique(probability)
            thresholds[index] = candidates[np.argmax([f1_score(truth, probability >= value) for value in candidates])]
        else:
            raise ValueError("method must be 'youden' or 'f1'")
    return np.clip(thresholds, 0, 1)


def apply_thresholds(probabilities: np.ndarray, thresholds: np.ndarray | float = 0.5) -> np.ndarray:
    return (np.asarray(probabilities) >= np.asarray(thresholds)).astype(np.int8)
