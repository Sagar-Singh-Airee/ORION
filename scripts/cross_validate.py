"""Create and persist leakage-safe folds; this does not train a model.

Usage:
    python make_folds.py --metadata metadata.csv --output folds.csv \\
        --group-column patient_id --n-folds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.text.label_extractor import FINDINGS  # noqa: E402
from orion.training.cross_validation import make_fold_assignments, save_fold_assignments  # noqa: E402

__all__ = ["main", "resolve_label_columns", "validate_group_column", "verify_no_group_leakage"]

FOLD_COLUMN = "fold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-column", default="patient_id")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_folds < 2:
        parser.error(f"--n-folds must be >= 2, got {args.n_folds}")
    return args


def resolve_label_columns(frame: pd.DataFrame) -> list[str]:
    """Return the FINDINGS columns present in the metadata, reporting what's missing."""
    labels = [label for label in FINDINGS if label in frame.columns]
    if not labels:
        raise ValueError("Metadata needs ORION label columns to stratify folds")
    missing = sorted(set(FINDINGS) - set(labels))
    if missing:
        print(f"Note: {len(missing)} FINDINGS column(s) absent from metadata and excluded from stratification: {missing}")
    return labels


def validate_group_column(frame: pd.DataFrame, group_column: str) -> None:
    """Fail fast on conditions that would silently defeat leakage-safe grouping."""
    if group_column not in frame.columns:
        raise ValueError(f"Metadata is missing required group column {group_column!r}")
    n_missing = frame[group_column].isna().sum()
    if n_missing:
        raise ValueError(
            f"{n_missing} row(s) have a null {group_column!r}; every row must belong to a "
            "group before folds can be assigned, or those rows will leak across folds"
        )


def verify_no_group_leakage(assigned: pd.DataFrame, group_column: str, fold_column: str = FOLD_COLUMN) -> None:
    """Confirm the one promise this script exists to keep: each group sits in exactly one fold."""
    if fold_column not in assigned.columns:
        raise ValueError(
            f"make_fold_assignments did not produce the expected {fold_column!r} column; "
            f"got columns {list(assigned.columns)}"
        )
    folds_per_group = assigned.groupby(group_column)[fold_column].nunique()
    leaking_groups = folds_per_group[folds_per_group > 1]
    if not leaking_groups.empty:
        raise RuntimeError(
            f"Leakage detected: {len(leaking_groups)} group(s) span multiple folds, e.g. "
            f"{leaking_groups.index[:5].tolist()}"
        )


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.metadata)
    if frame.empty:
        raise ValueError(f"{args.metadata} contains no rows")

    validate_group_column(frame, args.group_column)
    labels = resolve_label_columns(frame)

    assigned = make_fold_assignments(frame, args.group_column, labels, args.n_folds, args.seed)
    verify_no_group_leakage(assigned, args.group_column)

    output_path = save_fold_assignments(assigned, args.output)
    fold_sizes = assigned[FOLD_COLUMN].value_counts().sort_index()
    n_groups = assigned[args.group_column].nunique()
    print(f"Assigned {len(assigned)} row(s) across {n_groups} group(s) into {args.n_folds} folds -> {output_path}")
    print(f"Rows per fold: {fold_sizes.to_dict()}")


if __name__ == "__main__":
    main()