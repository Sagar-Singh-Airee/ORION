"""
Grouped Multi-Label Stratified Cross-Validation
================================================

WHY IT EXISTS
-------------
Medical imaging datasets frequently contain multiple studies, series, or
images belonging to the same patient.

A naive random split can cause patient leakage:

    Patient A -> training
    Patient A -> validation

The model may then learn patient-specific anatomy instead of pathology.

This module assigns complete patient/groups to folds while attempting to
balance multi-label targets across folds.

The splitter is deterministic and includes strict validation checks.

IMPORTANT
---------
This is a heuristic grouped stratification algorithm.

Its priorities are:

1. Never split a group across folds.
2. Preserve rare positive labels across folds when possible.
3. Keep label prevalence approximately balanced.
4. Keep fold sizes reasonably balanced.
5. Produce deterministic results for a fixed seed.

No stratification algorithm can guarantee perfect balance when rare labels
occur in very few groups.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .leakage import (
    assert_no_group_leakage,
    validate_fold_assignments,
)


__all__ = [
    "FoldStatistics",
    "iterative_group_kfold",
    "summarize_folds",
]


UNKNOWN_LABEL = -1


# ---------------------------------------------------------------------------
# Fold diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldStatistics:
    """Summary statistics for one validation fold."""

    fold: int
    n_samples: int
    n_groups: int
    label_counts: tuple[int, ...]
    label_prevalence: tuple[float, ...]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize splitter inputs."""

    labels = np.asarray(labels)
    groups = np.asarray(groups)

    if labels.ndim != 2:
        raise ValueError(
            f"labels must have shape (N, C), got {labels.shape}"
        )

    if groups.ndim != 1:
        raise ValueError(
            f"groups must have shape (N,), got {groups.shape}"
        )

    if len(labels) != len(groups):
        raise ValueError(
            "labels and groups must contain the same number of samples: "
            f"{len(labels)} != {len(groups)}"
        )

    if len(labels) == 0:
        raise ValueError("Cannot split an empty dataset")

    if labels.shape[1] == 0:
        raise ValueError("labels must contain at least one class")

    if not isinstance(n_splits, (int, np.integer)):
        raise TypeError(
            f"n_splits must be an integer, got {type(n_splits).__name__}"
        )

    if n_splits < 2:
        raise ValueError(
            f"n_splits must be at least 2, got {n_splits}"
        )

    if np.any(pd_is_nan(groups)):
        raise ValueError(
            "groups contain missing/NaN identifiers; "
            "patient-level leakage cannot be checked safely"
        )

    if np.any(~np.isfinite(
        labels.astype(float, copy=False)
    )):
        raise ValueError(
            "labels contain NaN or infinite values"
        )

    # Labels are expected to be:
    #
    #   1  = positive
    #   0  = negative
    #  -1  = unknown / missing
    #
    valid_values = np.isin(
        labels,
        [UNKNOWN_LABEL, 0, 1],
    )

    if not np.all(valid_values):
        invalid = np.unique(
            labels[~valid_values]
        )

        raise ValueError(
            "labels must contain only -1, 0, or 1; "
            f"found invalid values: {invalid.tolist()}"
        )

    groups = groups.astype(str)

    unique_groups = np.unique(groups)

    if len(unique_groups) < n_splits:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of unique groups "
            f"({len(unique_groups)})"
        )

    return labels.astype(np.int8, copy=False), groups


def pd_is_nan(values: np.ndarray) -> np.ndarray:
    """Return a boolean mask identifying NaN-like group identifiers."""

    try:
        numeric = values.astype(float)
    except (TypeError, ValueError):
        return np.zeros(len(values), dtype=bool)

    return np.isnan(numeric)


# ---------------------------------------------------------------------------
# Group aggregation
# ---------------------------------------------------------------------------


def _build_group_targets(
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Aggregate sample-level labels into group-level statistics.

    A group is considered positive for a class if ANY known sample belonging
    to that group is positive.

    Unknown labels (-1) are ignored.

    Returns
    -------
    unique_groups
    group_inverse
    group_targets
    group_sizes
    """

    unique_groups, group_inverse = np.unique(
        groups,
        return_inverse=True,
    )

    n_groups = len(unique_groups)
    n_classes = labels.shape[1]

    group_targets = np.zeros(
        (n_groups, n_classes),
        dtype=np.float64,
    )

    group_sizes = np.bincount(
        group_inverse,
        minlength=n_groups,
    ).astype(np.int64)

    for group_index in range(n_groups):
        rows = labels[group_inverse == group_index]

        known = rows != UNKNOWN_LABEL
        positives = rows == 1

        # A group is positive if any known sample is positive.
        group_targets[group_index] = (
            np.any(
                positives & known,
                axis=0,
            )
        ).astype(np.float64)

    return (
        unique_groups,
        group_inverse,
        group_targets,
        group_sizes,
    )


# ---------------------------------------------------------------------------
# Rarity weighting
# ---------------------------------------------------------------------------


def _compute_class_weights(
    group_targets: np.ndarray,
) -> np.ndarray:
    """
    Give greater importance to rare labels.

    A class occurring in fewer groups receives a larger balancing weight.
    """

    positive_counts = group_targets.sum(axis=0)

    weights = np.ones(
        group_targets.shape[1],
        dtype=np.float64,
    )

    for class_index, count in enumerate(positive_counts):
        if count <= 0:
            # Class does not occur at all.
            weights[class_index] = 0.0

        elif count <= 2:
            weights[class_index] = 4.0

        elif count <= 5:
            weights[class_index] = 3.0

        elif count <= 10:
            weights[class_index] = 2.0

        elif count <= 20:
            weights[class_index] = 1.5

        else:
            weights[class_index] = 1.0

    return weights


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------


def iterative_group_kfold(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create deterministic grouped multi-label cross-validation folds.

    Parameters
    ----------
    labels:
        Array with shape (N, C).

        Expected values:

            1  -> positive
            0  -> negative
           -1  -> unknown/missing

    groups:
        Array with shape (N,) containing patient/group identifiers.

    n_splits:
        Number of cross-validation folds.

    seed:
        Random seed used only for deterministic tie-breaking.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of (train_indices, validation_indices).

    Guarantees
    ---------
    A group is assigned to exactly one fold.

    Therefore:

        groups(train) ∩ groups(validation) = ∅

    for every fold.
    """

    labels, groups = _validate_inputs(
        labels,
        groups,
        n_splits,
    )

    (
        unique_groups,
        group_inverse,
        group_targets,
        group_sizes,
    ) = _build_group_targets(
        labels,
        groups,
    )

    n_groups = len(unique_groups)
    n_classes = labels.shape[1]

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------
    # Class rarity
    # ---------------------------------------------------------------

    class_weights = _compute_class_weights(
        group_targets
    )

    # ---------------------------------------------------------------
    # Desired distribution
    # ---------------------------------------------------------------

    total_targets = group_targets.sum(axis=0)

    desired_label_totals = (
        total_targets / n_splits
    )

    desired_group_size = (
        group_sizes.sum() / n_splits
    )

    # ---------------------------------------------------------------
    # Determine assignment order
    # ---------------------------------------------------------------
    #
    # Rare and information-rich groups are assigned first.
    #
    # This makes it harder for the first folds to consume all rare
    # positives.
    # ---------------------------------------------------------------

    rarity_score = (
        group_targets * class_weights
    ).sum(axis=1)

    label_count = group_targets.sum(axis=1)

    order_score = (
        rarity_score * 1000.0
        + label_count * 10.0
        + group_sizes.astype(float) * 0.001
    )

    # Randomized tie-breaking while remaining deterministic.
    random_tie_break = rng.random(n_groups)

    order = np.lexsort(
        (
            random_tie_break,
            -order_score,
        )
    )

    # ---------------------------------------------------------------
    # Fold state
    # ---------------------------------------------------------------

    fold_label_totals = np.zeros(
        (n_splits, n_classes),
        dtype=np.float64,
    )

    fold_group_counts = np.zeros(
        n_splits,
        dtype=np.int64,
    )

    fold_sample_counts = np.zeros(
        n_splits,
        dtype=np.int64,
    )

    assigned = np.full(
        n_groups,
        -1,
        dtype=np.int64,
    )

    # ---------------------------------------------------------------
    # Greedy iterative assignment
    # ---------------------------------------------------------------

    for group_index in order:
        target = group_targets[group_index]
        group_size = group_sizes[group_index]

        candidate_scores = np.zeros(
            n_splits,
            dtype=np.float64,
        )

        for fold in range(n_splits):

            proposed_labels = (
                fold_label_totals[fold]
                + target
            )

            # -------------------------------------------------------
            # Label-balance penalty
            # -------------------------------------------------------

            label_error = (
                proposed_labels
                - desired_label_totals
            )

            label_error = (
                label_error
                * class_weights
            )

            label_penalty = np.sum(
                label_error ** 2
            )

            # -------------------------------------------------------
            # Relative fold-size penalty
            # -------------------------------------------------------

            proposed_size = (
                fold_sample_counts[fold]
                + group_size
            )

            size_error = (
                proposed_size
                - desired_group_size
            ) / max(
                desired_group_size,
                1.0,
            )

            size_penalty = (
                size_error ** 2
            )

            # -------------------------------------------------------
            # Group-count penalty
            # -------------------------------------------------------

            expected_groups = (
                n_groups / n_splits
            )

            group_count_error = (
                fold_group_counts[fold] + 1
                - expected_groups
            ) / max(
                expected_groups,
                1.0,
            )

            group_penalty = (
                group_count_error ** 2
            )

            # -------------------------------------------------------
            # Combined objective
            # -------------------------------------------------------

            candidate_scores[fold] = (
                label_penalty
                + 0.25 * size_penalty
                + 0.05 * group_penalty
            )

        # Deterministic random tie-breaking.
        best_score = np.min(candidate_scores)

        candidates = np.flatnonzero(
            np.isclose(
                candidate_scores,
                best_score,
                rtol=1e-12,
                atol=1e-12,
            )
        )

        if len(candidates) == 1:
            chosen_fold = int(candidates[0])
        else:
            chosen_fold = int(
                rng.choice(candidates)
            )

        assigned[group_index] = chosen_fold

        fold_label_totals[chosen_fold] += target

        fold_group_counts[chosen_fold] += 1

        fold_sample_counts[chosen_fold] += group_size

    # ---------------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------------

    if np.any(assigned < 0):
        raise RuntimeError(
            "Internal splitter error: "
            "some groups were not assigned to a fold"
        )

    sample_folds = assigned[group_inverse]

    # ---------------------------------------------------------------
    # Validate assignments
    # ---------------------------------------------------------------

    validate_fold_assignments(
        groups,
        sample_folds,
    )

    # ---------------------------------------------------------------
    # Build CV splits
    # ---------------------------------------------------------------

    splits: list[
        tuple[np.ndarray, np.ndarray]
    ] = []

    for fold in range(n_splits):

        valid = np.flatnonzero(
            sample_folds == fold
        )

        train = np.flatnonzero(
            sample_folds != fold
        )

        if len(valid) == 0:
            raise RuntimeError(
                f"Fold {fold} contains no validation samples"
            )

        if len(train) == 0:
            raise RuntimeError(
                f"Fold {fold} contains no training samples"
            )

        # Absolute safety check against patient leakage.
        assert_no_group_leakage(
            groups[train],
            groups[valid],
        )

        splits.append(
            (
                train,
                valid,
            )
        )

    return splits


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def summarize_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    splits: Iterable[
        tuple[np.ndarray, np.ndarray]
    ],
) -> list[FoldStatistics]:
    """
    Produce diagnostic statistics for validation folds.

    This does NOT modify the folds.

    It is intended to make class imbalance and fold composition visible
    before training.
    """

    labels = np.asarray(labels)
    groups = np.asarray(groups).astype(str)

    if labels.ndim != 2:
        raise ValueError(
            f"labels must be 2-dimensional, got {labels.shape}"
        )

    summaries: list[FoldStatistics] = []

    for fold, (_, valid) in enumerate(splits):

        valid_labels = labels[valid]
        valid_groups = np.unique(
            groups[valid]
        )

        label_counts: list[int] = []
        prevalence: list[float] = []

        for class_index in range(
            labels.shape[1]
        ):

            class_labels = valid_labels[
                :,
                class_index,
            ]

            known = (
                class_labels != UNKNOWN_LABEL
            )

            positives = (
                class_labels == 1
            )

            count = int(
                np.sum(
                    positives & known
                )
            )

            denominator = int(
                np.sum(known)
            )

            label_counts.append(count)

            if denominator == 0:
                prevalence.append(float("nan"))
            else:
                prevalence.append(
                    count / denominator
                )

        summaries.append(
            FoldStatistics(
                fold=fold,
                n_samples=len(valid),
                n_groups=len(valid_groups),
                label_counts=tuple(label_counts),
                label_prevalence=tuple(prevalence),
            )
        )

    return summaries