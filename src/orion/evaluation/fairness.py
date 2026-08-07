"""Slice metrics by scanner/site metadata; never infer protected attributes."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .metrics import macro_auc


def group_macro_auc(targets: np.ndarray, probabilities: np.ndarray, groups: Sequence[object]) -> dict[str, float]:
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    if len(groups) != len(targets):
        raise ValueError("groups must align with targets")
    values: dict[str, float] = {}
    groups_array = np.asarray(groups).astype(str)
    for group in np.unique(groups_array):
        selected = groups_array == group
        values[group] = macro_auc(targets[selected], probabilities[selected])
    return values
