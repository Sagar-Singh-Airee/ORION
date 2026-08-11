"""
Pre-flight checks that prevent medically invalid validation splits.

WHY IT EXISTS
-------------

Medical imaging datasets frequently contain multiple studies, series, or
images belonging to the same patient.

A normal random split can therefore create data leakage:

    Patient A
        ├── Study 1 → training
        └── Study 2 → validation

The model may partially memorize patient-specific anatomy instead of learning
the underlying pathology.

This module provides strict checks for patient/group-level splitting.

A validation split should satisfy:

    groups(train) ∩ groups(valid) = ∅

and every group should belong to exactly one fold.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import isnan


__all__ = [
    "find_group_overlap",
    "assert_no_group_leakage",
    "validate_fold_assignments",
]


# ---------------------------------------------------------------------------
# Group normalization
# ---------------------------------------------------------------------------


def _normalize_group(value: object) -> str:
    """
    Normalize a patient/group identifier to a safe string.

    Missing identifiers are rejected because samples without a reliable group
    identifier cannot safely participate in patient-level splitting.

    Notes
    -----
    Group identifiers are commonly strings, integers, UUIDs, or other
    identifier-like objects. They are normalized to strings so that values
    such as 123 and "123" are treated consistently.
    """

    if value is None:
        raise ValueError(
            "Group identifier cannot be None; "
            "patient-level leakage cannot be checked safely."
        )

    # Check numeric NaN before converting to a string.
    #
    # This check is intentionally outside the try/except below so that the
    # ValueError raised for NaN cannot accidentally be swallowed.
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = None

    if numeric is not None and isnan(numeric):
        raise ValueError(
            "Group identifier cannot be NaN; "
            "patient-level leakage cannot be checked safely."
        )

    text = str(value).strip()

    if not text:
        raise ValueError(
            "Group identifier cannot be empty; "
            "patient-level leakage cannot be checked safely."
        )

    return text


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------


def find_group_overlap(
    train_groups: Iterable[object],
    valid_groups: Iterable[object],
) -> set[str]:
    """
    Return group identifiers present in both training and validation sets.

    Parameters
    ----------
    train_groups:
        Group/patient identifiers assigned to training.

    valid_groups:
        Group/patient identifiers assigned to validation.

    Returns
    -------
    set[str]
        Shared normalized identifiers.

    Notes
    -----
    Any non-empty result indicates potential patient/group leakage.
    """

    train = {
        _normalize_group(value)
        for value in train_groups
    }

    valid = {
        _normalize_group(value)
        for value in valid_groups
    }

    return train & valid


# ---------------------------------------------------------------------------
# Strict train/validation leakage assertion
# ---------------------------------------------------------------------------


def assert_no_group_leakage(
    train_groups: Iterable[object],
    valid_groups: Iterable[object],
) -> None:
    """
    Raise ValueError if any group occurs in both training and validation.

    This function should be called immediately after creating every
    patient-level validation split.
    """

    overlap = find_group_overlap(
        train_groups,
        valid_groups,
    )

    if not overlap:
        return

    preview = ", ".join(
        sorted(overlap)[:5]
    )

    if len(overlap) > 5:
        preview += ", ..."

    raise ValueError(
        "Patient/group leakage detected: "
        f"{len(overlap)} shared group IDs. "
        f"Examples: {preview}"
    )


# ---------------------------------------------------------------------------
# Fold-assignment validation
# ---------------------------------------------------------------------------


def validate_fold_assignments(
    groups: Iterable[object],
    folds: Iterable[int],
) -> None:
    """
    Ensure every group is assigned to exactly one fold.

    Parameters
    ----------
    groups:
        Group/patient identifier for each sample.

    folds:
        Fold assignment corresponding to each sample.

    Raises
    ------
    ValueError
        If a group appears in multiple folds, if a fold ID is invalid,
        or if group/fold lengths do not match.

    Notes
    -----
    This is particularly important when using GroupKFold-style validation.

    Example of invalid assignment:

        Patient A → fold 0
        Patient A → fold 1

    Even though the individual samples may look different, they belong to
    the same patient and therefore cannot safely occupy different folds.
    """

    group_to_fold: dict[str, int] = {}

    for group, fold in zip(
        groups,
        folds,
        strict=True,
    ):
        normalized_group = _normalize_group(group)

        try:
            fold_id = int(fold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid fold assignment {fold!r} "
                f"for group {normalized_group!r}"
            ) from exc

        if fold_id < 0:
            raise ValueError(
                f"Fold ID must be non-negative, got {fold_id} "
                f"for group {normalized_group!r}"
            )

        previous_fold = group_to_fold.get(
            normalized_group
        )

        if (
            previous_fold is not None
            and previous_fold != fold_id
        ):
            raise ValueError(
                f"Group {normalized_group!r} occurs in "
                f"multiple folds: {previous_fold} and {fold_id}"
            )

        group_to_fold[normalized_group] = fold_id

