"""Ensemble aligned OOF/test prediction CSVs.

Usage:
    python ensemble.py --predictions oof_a.csv oof_b.csv --output ensembled.csv \\
        --method rank_mean --weights 0.6 0.4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.text.label_extractor import FINDINGS  # noqa: E402
from orion.inference.ensemble import ensemble_predictions  # noqa: E402
from orion.inference.submission import save_submission  # noqa: E402

__all__ = ["main", "resolve_shared_labels", "validate_id_alignment", "validate_weights"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction CSV files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="study_id")
    parser.add_argument("--method", choices=("mean", "median", "rank_mean"), default="mean")
    parser.add_argument("--weights", type=float, nargs="*")
    return parser.parse_args()


def resolve_shared_labels(frames: list[pd.DataFrame], paths: list[str]) -> list[str]:
    """Return FINDINGS columns present in every prediction file.

    Reports which specific file is missing which label rather than silently
    ensembling over a shrunken label set.
    """
    missing_by_label = {
        label: [path for path, frame in zip(paths, frames) if label not in frame.columns] for label in FINDINGS
    }
    shared = [label for label, missing in missing_by_label.items() if not missing]
    partially_missing = {label: missing for label, missing in missing_by_label.items() if missing and len(missing) < len(frames)}
    if partially_missing:
        details = "; ".join(f"{label} missing from {missing}" for label, missing in partially_missing.items())
        raise ValueError(f"Label columns inconsistent across prediction files: {details}")
    if not shared:
        raise ValueError("Prediction files must share ORION label columns")
    return shared


def validate_weights(weights: list[float] | None, n_frames: int) -> None:
    if weights is None:
        return
    if len(weights) != n_frames:
        raise ValueError(f"--weights has {len(weights)} value(s) but {n_frames} prediction file(s) were given")
    if any(weight < 0 for weight in weights):
        raise ValueError("--weights must be non-negative")
    if not any(weight > 0 for weight in weights):
        raise ValueError("--weights must contain at least one positive value")


def check_for_nans(frames: list[pd.DataFrame], paths: list[str], labels: list[str]) -> None:
    for path, frame in zip(paths, frames):
        if frame[labels].isna().any().any():
            raise ValueError(f"{path} contains NaN values in label columns {labels}; cannot ensemble")


def validate_id_alignment(frames: list[pd.DataFrame], paths: list[str], id_column: str) -> pd.Series:
    """Confirm every file has the same id column and identical row order, naming the offender if not."""
    for path, frame in zip(paths, frames):
        if id_column not in frame.columns:
            raise ValueError(f"{path} is missing id column {id_column!r}")

    reference_path, reference_frame = paths[0], frames[0]
    reference_ids = reference_frame[id_column].astype(str).reset_index(drop=True)
    for path, frame in zip(paths[1:], frames[1:]):
        candidate_ids = frame[id_column].astype(str).reset_index(drop=True)
        if len(candidate_ids) != len(reference_ids):
            raise ValueError(
                f"{path} has {len(candidate_ids)} row(s) but {reference_path} has {len(reference_ids)}; "
                "prediction files must be row-aligned before ensembling"
            )
        if not reference_ids.equals(candidate_ids):
            raise ValueError(f"{path} has a different id order than {reference_path}; ensemble inputs must be row-aligned")
    return reference_ids


def main() -> None:
    args = parse_args()
    if len(args.predictions) == 1:
        print("Warning: only one prediction file provided; ensembling will be a no-op copy.")

    frames = [pd.read_csv(path) for path in args.predictions]
    for path, frame in zip(args.predictions, frames):
        if frame.empty:
            raise ValueError(f"{path} contains no rows")

    labels = resolve_shared_labels(frames, args.predictions)
    validate_weights(args.weights, len(frames))
    check_for_nans(frames, args.predictions, labels)
    ids = validate_id_alignment(frames, args.predictions, args.id_column)

    ensembled = ensemble_predictions([frame[labels].to_numpy() for frame in frames], args.weights, args.method)
    result = pd.DataFrame(ensembled, columns=labels)
    result.insert(0, args.id_column, ids)
    save_submission(result, args.output)

    print(
        f"Ensembled {len(frames)} file(s) via {args.method!r} over {len(labels)} label(s), "
        f"{len(result)} row(s) -> {args.output}"
    )


if __name__ == "__main__":
    main()