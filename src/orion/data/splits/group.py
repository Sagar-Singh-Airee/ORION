"""
Grouped Cross-Validation Splitting
==================================

WHY IT EXISTS
-------------

Medical imaging datasets commonly contain multiple studies, examinations,
or series belonging to the same patient.

A random K-Fold split can therefore create data leakage:

    Patient A / Study 2024 -> training
    Patient A / Study 2025 -> validation

The model may partially memorize patient-specific anatomy rather than learning
the underlying pathology.

This module provides a single entry point for patient/group-aware
cross-validation.

Supported strategies
--------------------

GroupKFold
    Strict group isolation without label stratification.

StratifiedGroupKFold
    Group isolation with single-label stratification.

iterative_group_kfold
    Group isolation with multi-label stratification.

All returned splits are passed through explicit leakage validation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

try:
    from sklearn.model_selection import (
        GroupKFold,
        StratifiedGroupKFold,
    )

    SKLEARN_AVAILABLE = True

except ImportError:
    GroupKFold = None
    StratifiedGroupKFold = None
    SKLEARN_AVAILABLE = False


from .leakage import (
    assert_no_group_leakage,
    validate_fold_assignments,
)


logger = logging.getLogger(__name__)


__all__ = [
    "get_group_kfold_splits",
]


# ---------------------------------------------------------------------------
# Group validation
# ---------------------------------------------------------------------------


def _validate_groups(
    groups: np.ndarray,
) -> np.ndarray:
    """
    Validate and normalize patient/group identifiers.

    Missing, empty, or NaN identifiers are rejected because patient-level
    leakage cannot be checked safely without a reliable group identifier.

    Returns
    -------
    np.ndarray
        Normalized string group identifiers.
    """

    groups = np.asarray(groups)

    if groups.ndim != 1:
        raise ValueError(
            f"groups must have shape (N,), got {groups.shape}"
        )

    if len(groups) == 0:
        raise ValueError(
            "groups cannot be empty"
        )

    normalized: list[str] = []

    for value in groups:

        if value is None:
            raise ValueError(
                "groups contain None patient/group identifiers"
            )

        # Detect numeric NaN without swallowing the resulting error.
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None

        if numeric is not None and np.isnan(numeric):
            raise ValueError(
                "groups contain NaN patient/group identifiers"
            )

        text = str(value).strip()

        if not text:
            raise ValueError(
                "groups contain empty patient/group identifiers"
            )

        normalized.append(text)

    return np.asarray(
        normalized,
        dtype=str,
    )


# ---------------------------------------------------------------------------
# Common input validation
# ---------------------------------------------------------------------------


def _validate_common_inputs(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate inputs shared by all splitting strategies.
    """

    X = np.asarray(X)
    y = np.asarray(y)
    groups = _validate_groups(groups)

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError(
            "X, y, and groups must contain the same number of samples: "
            f"X={len(X)}, y={len(y)}, groups={len(groups)}"
        )

    if len(X) == 0:
        raise ValueError(
            "Cannot create CV splits from an empty dataset"
        )

    if not isinstance(
        n_splits,
        (int, np.integer),
    ):
        raise TypeError(
            "n_splits must be an integer, "
            f"got {type(n_splits).__name__}"
        )

    n_splits = int(n_splits)

    if n_splits < 2:
        raise ValueError(
            f"n_splits must be at least 2, got {n_splits}"
        )

    n_groups = len(
        np.unique(groups)
    )

    if n_groups < n_splits:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number "
            f"of unique groups={n_groups}"
        )

    return X, y, groups


# ---------------------------------------------------------------------------
# Split validation
# ---------------------------------------------------------------------------


def _validate_splits(
    splits: Sequence[
        tuple[np.ndarray, np.ndarray]
    ],
    groups: np.ndarray,
    n_splits: int,
) -> list[
    tuple[np.ndarray, np.ndarray]
]:
    """
    Perform strict post-split validation.

    This protects the pipeline even when the underlying splitter is external.

    Checks include:

    - Correct number of folds.
    - Non-empty training sets.
    - Non-empty validation sets.
    - Valid sample indices.
    - No sample in both train and validation.
    - No patient/group leakage.
    - Every sample appears in exactly one validation fold.
    - Every group belongs to exactly one validation fold.
    """

    if len(splits) != n_splits:
        raise RuntimeError(
            f"Expected {n_splits} folds, "
            f"got {len(splits)}"
        )

    sample_fold_ids = np.full(
        len(groups),
        -1,
        dtype=np.int64,
    )

    validated: list[
        tuple[np.ndarray, np.ndarray]
    ] = []

    for fold, (train, valid) in enumerate(splits):

        train = np.asarray(
            train,
            dtype=np.int64,
        )

        valid = np.asarray(
            valid,
            dtype=np.int64,
        )

        # ---------------------------------------------------------------
        # Basic fold checks
        # ---------------------------------------------------------------

        if len(train) == 0:
            raise RuntimeError(
                f"Fold {fold} has an empty training set"
            )

        if len(valid) == 0:
            raise RuntimeError(
                f"Fold {fold} has an empty validation set"
            )

        # ---------------------------------------------------------------
        # Index bounds
        # ---------------------------------------------------------------

        if np.any(train < 0) or np.any(
            train >= len(groups)
        ):
            raise RuntimeError(
                f"Fold {fold} contains invalid training indices"
            )

        if np.any(valid < 0) or np.any(
            valid >= len(groups)
        ):
            raise RuntimeError(
                f"Fold {fold} contains invalid validation indices"
            )

        # ---------------------------------------------------------------
        # Sample-level overlap
        # ---------------------------------------------------------------

        if np.intersect1d(
            train,
            valid,
        ).size:
            raise RuntimeError(
                f"Fold {fold} contains samples in both "
                "training and validation"
            )

        # ---------------------------------------------------------------
        # Patient/group leakage
        # ---------------------------------------------------------------

        assert_no_group_leakage(
            groups[train],
            groups[valid],
        )

        # ---------------------------------------------------------------
        # Every validation sample must belong to exactly one fold.
        # ---------------------------------------------------------------

        for index in valid:

            if sample_fold_ids[index] != -1:
                raise RuntimeError(
                    f"Sample {index} appears in multiple "
                    "validation folds"
                )

            sample_fold_ids[index] = fold

        validated.append(
            (
                train,
                valid,
            )
        )

    # -------------------------------------------------------------------
    # Every sample must appear in exactly one validation fold.
    # -------------------------------------------------------------------

    if np.any(sample_fold_ids == -1):

        missing = np.flatnonzero(
            sample_fold_ids == -1
        )

        raise RuntimeError(
            f"{len(missing)} samples were not assigned "
            "to any validation fold"
        )

    # -------------------------------------------------------------------
    # Every group must belong to exactly one validation fold.
    # -------------------------------------------------------------------

    validate_fold_assignments(
        groups,
        sample_fold_ids,
    )

    return validated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_group_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    stratified: bool = False,
    seed: int = 42,
) -> list[
    tuple[np.ndarray, np.ndarray]
]:
    """
    Generate patient/group-aware cross-validation splits.

    Parameters
    ----------
    X:
        Feature array or sample indices.

    y:
        Labels.

        Supported forms:

            (N,)
                Single-label classification.

            (N, C)
                Multi-label classification.

    groups:
        Patient/group identifier for every sample.

    n_splits:
        Number of cross-validation folds.

    stratified:
        If False:
            Use GroupKFold.

        If True and y is one-dimensional:
            Use scikit-learn StratifiedGroupKFold.

        If True and y is two-dimensional:
            Use the project's deterministic multi-label
            iterative_group_kfold implementation.

    seed:
        Random seed controlling deterministic stratified splitting.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Training and validation indices for every fold.

    Notes
    -----
    Patient/group isolation is always enforced.

    Multi-label targets are never flattened into a fake single class.
    """

    X, y, groups = _validate_common_inputs(
        X,
        y,
        groups,
        n_splits,
    )

    # -------------------------------------------------------------------
    # Multi-label stratification
    # -------------------------------------------------------------------

    if stratified and y.ndim == 2:

        logger.info(
            "Using deterministic multi-label grouped "
            "stratification with %d folds.",
            n_splits,
        )

        from .stratified import (
            iterative_group_kfold,
        )

        splits = iterative_group_kfold(
            labels=y,
            groups=groups,
            n_splits=n_splits,
            seed=seed,
        )

    # -------------------------------------------------------------------
    # Single-label stratification
    # -------------------------------------------------------------------

    elif stratified:

        if y.ndim != 1:
            raise ValueError(
                "Single-label StratifiedGroupKFold expects "
                "y with shape (N,)"
            )

        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for "
                "StratifiedGroupKFold."
            )

        logger.info(
            "Using StratifiedGroupKFold with "
            "%d folds and seed=%d.",
            n_splits,
            seed,
        )

        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        splits = list(
            cv.split(
                X,
                y,
                groups=groups,
            )
        )

    # -------------------------------------------------------------------
    # Standard grouped CV
    # -------------------------------------------------------------------

    else:

        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for "
                "GroupKFold."
            )

        logger.info(
            "Using GroupKFold with %d folds.",
            n_splits,
        )

        cv = GroupKFold(
            n_splits=n_splits,
        )

        splits = list(
            cv.split(
                X,
                y,
                groups=groups,
            )
        )

    # -------------------------------------------------------------------
    # Final safety validation
    # -------------------------------------------------------------------

    return _validate_splits(
        splits,
        groups,
        n_splits,
    )

