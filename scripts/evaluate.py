"""Evaluate OOF predictions; all metric inputs are aligned by study ID.

Usage:
    python evaluate.py --targets targets.csv --predictions oof_preds.csv --output metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.text.label_extractor import FINDINGS  # noqa: E402
from orion.evaluation.metrics import calculate_metrics  # noqa: E402

__all__ = ["main", "resolve_shared_labels", "align_by_id", "align_by_position", "validate_arrays"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--id-column", default="study_id")
    parser.add_argument("--output")
    parser.add_argument(
        "--assume-row-aligned",
        action="store_true",
        help=(
            "Evaluate by row position when --id-column is absent from one or both files. "
            "Dangerous: silently wrong if the two files' row order differs. Off by default."
        ),
    )
    return parser.parse_args()


def resolve_shared_labels(targets: pd.DataFrame, predictions: pd.DataFrame) -> list[str]:
    """Return FINDINGS columns present in both files, naming whichever file is missing which label."""
    frames = {"targets": targets, "predictions": predictions}
    missing_by_label = {
        label: [name for name, df in frames.items() if label not in df.columns] for label in FINDINGS
    }
    shared = [label for label, missing in missing_by_label.items() if not missing]
    partially_missing = {label: missing for label, missing in missing_by_label.items() if missing and len(missing) < 2}
    if partially_missing:
        details = "; ".join(f"{label} missing from {missing}" for label, missing in partially_missing.items())
        raise ValueError(f"Label columns inconsistent between targets and predictions: {details}")
    if not shared:
        raise ValueError("No shared target columns; expected canonical ORION labels")
    return shared


def align_by_id(
    targets: pd.DataFrame, predictions: pd.DataFrame, labels: list[str], id_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Merge targets and predictions on id_column, requiring a complete, exact-match alignment."""
    merged = targets[[id_column, *labels]].merge(
        predictions[[id_column, *labels]], on=id_column, suffixes=("_target", "_prediction"), validate="one_to_one"
    )
    if len(merged) != len(targets) or len(merged) != len(predictions):
        raise ValueError(
            f"Targets and predictions ids must match exactly before evaluation "
            f"(targets={len(targets)}, predictions={len(predictions)}, matched={len(merged)})"
        )
    truth = merged[[f"{label}_target" for label in labels]].to_numpy()
    probability = merged[[f"{label}_prediction" for label in labels]].to_numpy()
    return truth, probability


def align_by_position(targets: pd.DataFrame, predictions: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fallback alignment with no id column to verify against — caller must have opted in explicitly."""
    if len(targets) != len(predictions):
        raise ValueError(
            f"Rows differ (targets={len(targets)}, predictions={len(predictions)}) and no shared id column exists"
        )
    print(
        "WARNING: evaluating by row position, not study id. Metrics will be silently wrong "
        "if the two files are not in identical row order."
    )
    return targets[labels].to_numpy(), predictions[labels].to_numpy()


def validate_arrays(truth: np.ndarray, probability: np.ndarray, labels: list[str]) -> None:
    if np.isnan(truth).any():
        raise ValueError(f"Targets contain {int(np.isnan(truth).sum())} NaN value(s) across columns {labels}")
    if np.isnan(probability).any():
        raise ValueError(f"Predictions contain {int(np.isnan(probability).sum())} NaN value(s) across columns {labels}")
    if probability.min() < 0 or probability.max() > 1:
        print(
            f"Note: predicted probabilities fall outside [0, 1] (min={probability.min():.4f}, "
            f"max={probability.max():.4f}); AUC is unaffected by monotonic scaling but check this is intentional."
        )
    if truth.min() < 0 or truth.max() > 1:
        print(f"Note: target values fall outside [0, 1] (min={truth.min():.4f}, max={truth.max():.4f})")


def main() -> None:
    args = parse_args()
    targets = pd.read_csv(args.targets)
    predictions = pd.read_csv(args.predictions)
    for name, df in (("targets", targets), ("predictions", predictions)):
        if df.empty:
            raise ValueError(f"{name} file contains no rows")

    labels = resolve_shared_labels(targets, predictions)
    id_available = args.id_column in targets.columns and args.id_column in predictions.columns

    if id_available:
        truth, probability = align_by_id(targets, predictions, labels, args.id_column)
    elif args.assume_row_aligned:
        truth, probability = align_by_position(targets, predictions, labels)
    else:
        raise ValueError(
            f"id column {args.id_column!r} is missing from targets and/or predictions. "
            "Add it, or pass --assume-row-aligned to explicitly evaluate by row position instead."
        )

    validate_arrays(truth, probability, labels)
    metrics = calculate_metrics(truth, probability, labels)
    rendered = json.dumps(metrics, indent=2, allow_nan=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")

    print(rendered)
    alignment = "id-aligned" if id_available else "row-position aligned"
    print(f"Evaluated {len(labels)} label(s) over {len(truth)} row(s) ({alignment})")


if __name__ == "__main__":
    main()