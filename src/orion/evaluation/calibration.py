"""Probability calibration and diagnostics without test-set leakage."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), eps, 1 - eps)
    return np.log(probabilities / (1 - probabilities))


class TemperatureScaler:
    """One temperature per task, fit on held-out out-of-fold predictions only."""

    def __init__(self, temperatures: np.ndarray | None = None):
        self.temperatures = temperatures

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> "TemperatureScaler":
        probabilities, targets = np.asarray(probabilities), np.asarray(targets)
        if probabilities.shape != targets.shape or probabilities.ndim != 2:
            raise ValueError("probabilities and targets must be same (N, C) shape")
        temperatures = np.ones(probabilities.shape[1], dtype=float)
        logits = _logit(probabilities)
        for index in range(probabilities.shape[1]):
            known = targets[:, index] >= 0
            target = targets[known, index]
            if len(target) == 0 or np.unique(target).size < 2:
                continue
            def nll(temperature: float) -> float:
                scaled = logits[known, index] / temperature
                return float(np.mean(np.logaddexp(0, scaled) - target * scaled))
            temperatures[index] = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x
        self.temperatures = temperatures
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.temperatures is None:
            raise RuntimeError("Call fit before transform")
        logits = _logit(probabilities)
        return (1 / (1 + np.exp(-logits / self.temperatures))).astype(np.float32)


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    probabilities, targets = np.asarray(probabilities).ravel(), np.asarray(targets).ravel()
    known = targets >= 0
    probabilities, targets = probabilities[known], targets[known]
    if not len(targets):
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if mask.any():
            ece += mask.mean() * abs(probabilities[mask].mean() - targets[mask].mean())
    return float(ece)
