"""Prediction ensembling with shape and weight validation."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def ensemble_predictions(predictions: Sequence[np.ndarray], weights: Sequence[float] | None = None, method: str = "mean") -> np.ndarray:
    if not predictions:
        raise ValueError("At least one prediction array is required")
    arrays = [np.asarray(prediction, dtype=np.float64) for prediction in predictions]
    shape = arrays[0].shape
    if len(shape) != 2:
        raise ValueError("Each prediction array must have shape (N, C)")
    if any(array.shape != shape for array in arrays):
        raise ValueError("All prediction arrays must have the same shape")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Predictions contain NaN or infinite values")
    stack = np.stack(arrays, axis=0)
    if method == "mean":
        if weights is None:
            return stack.mean(axis=0).astype(np.float32)
        weights_array = np.asarray(weights, dtype=float)
        if weights_array.shape != (len(arrays),) or np.any(weights_array < 0) or weights_array.sum() <= 0:
            raise ValueError("weights must be non-negative and align with predictions")
        return np.average(stack, axis=0, weights=weights_array).astype(np.float32)
    if method == "median":
        return np.median(stack, axis=0).astype(np.float32)
    if method == "rank_mean":
        ranks = np.empty_like(stack)
        for model in range(len(arrays)):
            for label in range(shape[1]):
                ranks[model, :, label] = np.argsort(np.argsort(stack[model, :, label])) / max(1, shape[0] - 1)
        return ranks.mean(axis=0).astype(np.float32)
    raise ValueError("method must be mean, median, or rank_mean")
