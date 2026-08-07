"""Pre-flight checks that prevent medically invalid validation splits."""
from __future__ import annotations

from collections.abc import Iterable


def find_group_overlap(train_groups: Iterable[object], valid_groups: Iterable[object]) -> set[str]:
    """Return shared identifiers, normalized to strings for safe CSV interoperability."""
    return {str(value) for value in train_groups} & {str(value) for value in valid_groups}


def assert_no_group_leakage(train_groups: Iterable[object], valid_groups: Iterable[object]) -> None:
    overlap = find_group_overlap(train_groups, valid_groups)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Patient/group leakage detected ({len(overlap)} shared IDs): {preview}")


def validate_fold_assignments(groups: Iterable[object], folds: Iterable[int]) -> None:
    """Ensure all samples belonging to a patient have exactly one assigned fold."""
    group_to_fold: dict[str, int] = {}
    for group, fold in zip(groups, folds, strict=True):
        group = str(group)
        if group in group_to_fold and group_to_fold[group] != int(fold):
            raise ValueError(f"Group {group!r} occurs in multiple folds")
        group_to_fold[group] = int(fold)
