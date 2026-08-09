"""Fit classwise temperature scaling on OOF predictions and apply it to test predictions.

Usage:
    python calibrate.py --targets oof_targets.csv --oof oof_preds.csv \\
        --test test_preds.csv --output test_preds_calibrated.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.text.label_extractor import FINDINGS  # noqa: E402
from orion.evaluation.calibration import TemperatureScaler  # noqa: E402

__all__ = ["main", "resolve_shared_labels", "build_calibration_arrays", "calibrate_test_predictions"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, help="OOF ground-truth CSV")
    parser.add_argument("--oof", required=True, help="OOF probability CSV")
    parser.add_argument("--test", required=True, help="Test probability CSV")
    parser.add_argument("--output", required=True, help="Where to write calibrated test predictions")
    parser.add_argument("--id-column", default="study_id")
    return parser.parse_args()


def resolve_shared_labels(
    targets: pd.DataFrame, oof: pd.DataFrame, test: pd.DataFrame
) -> list[str]:
    """Return the FINDINGS columns present in all three frames, or fail with a precise reason.

    Rather than silently dropping labels missing from any one file, this reports exactly
    which file is missing which label so a schema mismatch is caught immediately.
    """
    frames = {"targets": targets, "oof": oof, "test": test}
    missing_by_label = {
        label: [name for name, df in frames.items() if label not in df.columns]
        for label in FINDINGS
    }
    shared = [label for label, missing in missing_by_label.items() if not missing]
    partially_missing = {label: missing for label, missing in missing_by_label.items() if missing and len(missing) < 3}
    if partially_missing:
        details = "; ".join(f"{label} missing from {missing}" for label, missing in partially_missing.items())
        raise ValueError(f"Label columns inconsistent across inputs: {details}")
    if not shared:
        raise ValueError(
            f"None of the ORION FINDINGS columns {FINDINGS} were found in all of "
            "targets, OOF predictions, and test predictions."
        )
    return shared


def require_id_column(df: pd.DataFrame, id_column: str, name: str) -> None:
    if id_column not in df.columns:
        raise ValueError(f"{name} is missing required id column {id_column!r}")


def build_calibration_arrays(
    targets: pd.DataFrame, oof: pd.DataFrame, labels: list[str], id_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Align targets and OOF predictions row-for-row and return (oof_array, target_array)."""
    merged = targets[[id_column, *labels]].merge(
        oof[[id_column, *labels]],
        on=id_column,
        suffixes=("_target", "_prediction"),
        validate="one_to_one",
    )
    if len(merged) != len(targets) or len(merged) != len(oof):
        raise ValueError(
            f"OOF target and prediction ids must match exactly before calibration "
            f"(targets={len(targets)}, oof={len(oof)}, matched={len(merged)})"
        )
    target_array = merged[[f"{label}_target" for label in labels]].to_numpy()
    oof_array = merged[[f"{label}_prediction" for label in labels]].to_numpy()
    for name, array in (("targets", target_array), ("OOF predictions", oof_array)):
        if np.isnan(array).any():
            raise ValueError(f"{name} contain NaN values for columns {labels}; cannot fit calibration")
    return oof_array, target_array


def calibrate_test_predictions(
    oof_array: np.ndarray, target_array: np.ndarray, test: pd.DataFrame, labels: list[str]
) -> pd.DataFrame:
    """Fit a TemperatureScaler on OOF data and apply it to the test prediction columns."""
    test_array = test[labels].to_numpy()
    if np.isnan(test_array).any():
        raise ValueError(f"Test predictions contain NaN values for columns {labels}; cannot calibrate")
    scaler = TemperatureScaler().fit(oof_array, target_array)
    calibrated = test.copy()
    calibrated.loc[:, labels] = scaler.transform(test_array)
    return calibrated


def main() -> None:
    args = parse_args()
    targets = pd.read_csv(args.targets)
    oof = pd.read_csv(args.oof)
    test = pd.read_csv(args.test)

    for name, df in (("targets", targets), ("OOF predictions", oof), ("test predictions", test)):
        require_id_column(df, args.id_column, name)

    labels = resolve_shared_labels(targets, oof, test)
    oof_array, target_array = build_calibration_arrays(targets, oof, labels, args.id_column)
    calibrated = calibrate_test_predictions(oof_array, target_array, test, labels)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(output, index=False, float_format="%.7f")
    print(f"Calibrated {len(labels)} label(s) on {len(oof_array)} OOF rows -> {output}")
    print(f"Labels: {labels}")


if __name__ == "__main__":
    main()