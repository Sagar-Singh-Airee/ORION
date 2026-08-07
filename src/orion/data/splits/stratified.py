"""Multi-label-aware grouped CV splitting with a deterministic fallback."""
from __future__ import annotations

import numpy as np

from .leakage import assert_no_group_leakage


def iterative_group_kfold(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Assign whole patients to folds while approximately balancing label prevalence."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if labels.ndim != 2 or len(labels) != len(groups):
        raise ValueError("labels must be (N, C) and align with groups")
    unique_groups, group_inverse = np.unique(groups.astype(str), return_inverse=True)
    if len(unique_groups) < n_splits:
        raise ValueError("n_splits cannot exceed the number of unique groups")

    group_targets = np.zeros((len(unique_groups), labels.shape[1]), dtype=float)
    for index in range(len(unique_groups)):
        rows = labels[group_inverse == index]
        group_targets[index] = np.where(rows >= 0, rows, 0).sum(axis=0)

    rng = np.random.default_rng(seed)
    order = np.arange(len(unique_groups))
    rng.shuffle(order)
    order = order[np.argsort(-group_targets[order].sum(axis=1), kind="stable")]
    desired = group_targets.sum(axis=0) / n_splits
    fold_totals = np.zeros((n_splits, labels.shape[1]), dtype=float)
    fold_sizes = np.zeros(n_splits, dtype=int)
    assigned = np.empty(len(unique_groups), dtype=int)
    for group_index in order:
        target = group_targets[group_index]
        scores = ((fold_totals + target - desired) ** 2).sum(axis=1)
        scores += fold_sizes / max(1, len(groups))
        fold = int(np.argmin(scores))
        assigned[group_index] = fold
        fold_totals[fold] += target
        fold_sizes[fold] += int((group_inverse == group_index).sum())

    sample_folds = assigned[group_inverse]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        valid = np.flatnonzero(sample_folds == fold)
        train = np.flatnonzero(sample_folds != fold)
        assert_no_group_leakage(groups[train], groups[valid])
        splits.append((train, valid))
    return splits
