"""Validate an existing prediction CSV and write a Kaggle-ready submission CSV.

Usage:
    python submit.py --predictions oof_preds.csv --output submission.csv \\
        --sample-submission sample_submission.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.inference.submission import create_submission, save_submission  # noqa: E402

__all__ = ["main", "resolve_labels", "align_to_sample_order", "validate_no_nans"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="study_id")
    parser.add_argument("--sample-submission", help="Use its exact label order AND row order (required by Kaggle scoring)")
    return parser.parse_args()


def load_frame(path: str, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{name} file ({path}) contains no rows")
    return frame


def resolve_labels(predictions: pd.DataFrame, id_column: str, sample: pd.DataFrame | None) -> list[str]:
    """Decide the label column order, and confirm predictions actually has every label required."""
    if sample is not None:
        if id_column not in sample.columns:
            raise ValueError(f"--sample-submission is missing id column {id_column!r}; found {list(sample.columns)}")
        labels = [column for column in sample.columns if column != id_column]
    else:
        labels = [column for column in predictions.columns if column != id_column]

    if not labels:
        raise ValueError("No label columns resolved; the predictions/sample-submission file has only the id column")

    missing = [label for label in labels if label not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions file is missing label column(s) required by the submission format: {missing}")
    return labels


def _require_unique_ids(frame: pd.DataFrame, id_column: str, name: str) -> None:
    duplicate_count = int(frame[id_column].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"{name} has {duplicate_count} duplicate {id_column!r} value(s)")


def align_to_sample_order(
    predictions: pd.DataFrame, sample: pd.DataFrame, id_column: str, labels: list[str]
) -> pd.DataFrame:
    """Reindex predictions to the sample submission's exact row order.

    Kaggle scoring depends on submission rows matching the sample's id order exactly —
    matching only the *column* order (as the original script did) is not sufficient.
    """
    _require_unique_ids(predictions, id_column, "predictions")
    _require_unique_ids(sample, id_column, "sample submission")

    aligned = sample[[id_column]].merge(
        predictions[[id_column, *labels]], on=id_column, how="left", validate="one_to_one"
    )
    missing_mask = aligned[labels].isna().any(axis=1)
    if missing_mask.any():
        missing_ids = aligned.loc[missing_mask, id_column].tolist()
        raise ValueError(
            f"{len(missing_ids)} id(s) required by --sample-submission have no matching prediction, "
            f"e.g. {missing_ids[:5]}"
        )

    extra_ids = set(predictions[id_column]) - set(sample[id_column])
    if extra_ids:
        print(f"Note: {len(extra_ids)} prediction id(s) not present in --sample-submission were dropped")
    return aligned


def validate_no_nans(frame: pd.DataFrame, labels: list[str]) -> None:
    if frame[labels].isna().any().any():
        raise ValueError(f"Prediction values contain NaN in columns {labels}; cannot create a valid submission")


def build_submission_frame(ids: pd.Series, values: pd.DataFrame, labels: list[str], id_column: str) -> pd.DataFrame:
    result = create_submission(ids, values.to_numpy(), labels)
    if id_column != "study_id" and "study_id" in result.columns:
        result = result.rename(columns={"study_id": id_column})
    return result


def main() -> None:
    args = parse_args()
    predictions = load_frame(args.predictions, "predictions")
    if args.id_column not in predictions.columns:
        raise ValueError(f"Missing ID column {args.id_column!r} in predictions; found {list(predictions.columns)}")

    sample = load_frame(args.sample_submission, "sample submission") if args.sample_submission else None
    labels = resolve_labels(predictions, args.id_column, sample)
    validate_no_nans(predictions, labels)

    if sample is not None:
        aligned = align_to_sample_order(predictions, sample, args.id_column, labels)
        ids, values = aligned[args.id_column], aligned[labels]
    else:
        _require_unique_ids(predictions, args.id_column, "predictions")
        ids, values = predictions[args.id_column], predictions[labels]

    result = build_submission_frame(ids, values, labels, args.id_column)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_submission(result, args.output)

    alignment_note = " (row order matched to --sample-submission)" if sample is not None else ""
    print(f"Wrote {len(result)} row(s), {len(labels)} label(s) -> {output_path}{alignment_note}")


if __name__ == "__main__":
    main()