"""Competition metrics with explicit handling of missing weak labels."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if y_true.ndim != 2 or y_prob.ndim != 2 or y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must be equally shaped (N, C) arrays")
    if not np.isfinite(y_prob).all():
        raise ValueError("Predictions contain NaN or infinite values")
    return y_true, np.clip(y_prob.astype(float), 0.0, 1.0)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    label_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Calculate macro ROC-AUC/AP plus per-label values.

    Targets below zero are treated as abstentions, so report-derived weak labels do
    not turn missing mention into a false negative. Single-class folds are reported
    as NaN and excluded from macro values rather than quietly assigned an AUC of 0.
    """
    y_true, y_prob = _validate(y_true, y_pred_probs)
    labels = list(label_names or [f"class_{index}" for index in range(y_true.shape[1])])
    if len(labels) != y_true.shape[1]:
        raise ValueError("label_names must have one entry per target")
    metrics: dict[str, float] = {}
    aucs: list[float] = []
    aps: list[float] = []
    for index, label in enumerate(labels):
        known = y_true[:, index] >= 0
        truth, prob = y_true[known, index], y_prob[known, index]
        if len(truth) == 0 or np.unique(truth).size < 2:
            metrics[f"auc_{label}"] = float("nan")
            metrics[f"ap_{label}"] = float("nan")
            continue
        auc = float(roc_auc_score(truth, prob))
        ap = float(average_precision_score(truth, prob))
        metrics[f"auc_{label}"] = auc
        metrics[f"ap_{label}"] = ap
        metrics[f"brier_{label}"] = float(brier_score_loss(truth, prob))
        aucs.append(auc)
        aps.append(ap)
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    metrics["macro_auc"] = macro_auc
    metrics["macro_auc_roc"] = macro_auc
    metrics["macro_average_precision"] = float(np.mean(aps)) if aps else float("nan")
    return metrics


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return calculate_metrics(y_true, y_prob)["macro_auc_roc"]
