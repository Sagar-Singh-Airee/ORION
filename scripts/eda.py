"""Emit a compact, reproducible data audit before selecting an experiment.

Usage:
    python eda.py --metadata metadata.csv --group-column patient_id --output audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.text.label_extractor import FINDINGS  # noqa: E402

__all__ = ["main", "compute_column_summary", "compute_label_summary", "compute_group_summary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", help="Optional path to also write the audit as JSON")
    parser.add_argument("--group-column", help="e.g. patient_id — reports group cardinality and leakage-relevant sizing")
    parser.add_argument(
        "--label-columns",
        nargs="*",
        help="Override which columns to treat as findings; defaults to whichever ORION FINDINGS are present",
    )
    return parser.parse_args()


def compute_column_summary(frame: pd.DataFrame) -> dict:
    """Shape, dtypes, and missingness — sorted worst-first so problems are visible immediately."""
    missing_fraction = frame.isna().mean().round(4).sort_values(ascending=False)
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing_fraction": missing_fraction.to_dict(),
        "unique_counts": frame.nunique(dropna=True).to_dict(),
    }


def compute_label_summary(frame: pd.DataFrame, label_columns: list[str]) -> dict:
    """Per-label prevalence — the single most decision-relevant fact for this competition's
    class-imbalanced, multi-label setup. Works for hard (0/1) or soft (weak-label) targets.
    """
    summary = {}
    for label in label_columns:
        column = pd.to_numeric(frame[label], errors="coerce")
        n_present = int(column.notna().sum())
        summary[label] = {
            "n_present": n_present,
            "n_missing": int(len(frame) - n_present),
            "mean": round(float(column.mean()), 4) if n_present else None,
            "min": round(float(column.min()), 4) if n_present else None,
            "max": round(float(column.max()), 4) if n_present else None,
        }
    return summary


def compute_group_summary(frame: pd.DataFrame, group_column: str) -> dict:
    """Group cardinality and rows-per-group spread — surfaces leakage-relevant sizing early,
    before a CV strategy (see cross_validate.py) is chosen.
    """
    if group_column not in frame.columns:
        raise ValueError(f"--group-column {group_column!r} is not a column in the metadata")
    n_missing = int(frame[group_column].isna().sum())
    group_sizes = frame[group_column].value_counts(dropna=True)
    return {
        "n_groups": int(group_sizes.size),
        "n_rows_missing_group": n_missing,
        "rows_per_group": {
            "min": int(group_sizes.min()) if not group_sizes.empty else None,
            "median": float(group_sizes.median()) if not group_sizes.empty else None,
            "max": int(group_sizes.max()) if not group_sizes.empty else None,
        },
    }


def build_summary(frame: pd.DataFrame, label_columns: list[str] | None, group_column: str | None) -> dict:
    resolved_labels = label_columns if label_columns is not None else [f for f in FINDINGS if f in frame.columns]
    summary = {"columns": compute_column_summary(frame)}

    if resolved_labels:
        summary["labels"] = compute_label_summary(frame, resolved_labels)
    else:
        summary["labels"] = {"note": "no ORION FINDINGS columns found; pass --label-columns to audit specific columns"}

    if group_column:
        summary["groups"] = compute_group_summary(frame, group_column)

    return summary


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.metadata)
    if frame.empty:
        raise ValueError(f"{args.metadata} contains no rows")

    summary = build_summary(frame, args.label_columns, args.group_column)
    rendered = json.dumps(summary, indent=2, default=str)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")

    print(rendered)


if __name__ == "__main__":
    main()