"""Structured error tables for failure-mode review."""
from __future__ import annotations

import numpy as np
import pandas as pd


def prediction_error_table(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    study_ids: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    if targets.shape != probabilities.shape or targets.shape[1] != len(label_names):
        raise ValueError("Target, prediction and label shapes do not agree")
    rows = []
    for sample, label in zip(*np.where(targets >= 0), strict=True):
        truth, probability = int(targets[sample, label]), float(probabilities[sample, label])
        rows.append({"study_uid": study_ids[sample] if study_ids else str(sample), "label": label_names[label], "target": truth, "probability": probability, "prediction": int(probability >= threshold), "absolute_error": abs(truth - probability)})
    return pd.DataFrame(rows).sort_values("absolute_error", ascending=False, ignore_index=True)
