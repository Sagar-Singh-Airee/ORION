"""Fit classwise temperature scaling on OOF predictions and apply it to test predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.text.label_extractor import FINDINGS
from orion.evaluation.calibration import TemperatureScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, help="OOF ground-truth CSV")
    parser.add_argument("--oof", required=True, help="OOF probability CSV")
    parser.add_argument("--test", required=True, help="Test probability CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="study_id")
    args = parser.parse_args()

    targets, oof, test = pd.read_csv(args.targets), pd.read_csv(args.oof), pd.read_csv(args.test)
    labels = [label for label in FINDINGS if label in targets and label in oof and label in test]
    if not labels:
        raise ValueError("Targets, OOF predictions, and test predictions must share ORION label columns")
    if args.id_column not in targets or args.id_column not in oof or args.id_column not in test:
        raise ValueError(f"Targets, OOF predictions, and test predictions must contain {args.id_column!r}")
    merged = targets[[args.id_column, *labels]].merge(
        oof[[args.id_column, *labels]], on=args.id_column, suffixes=("_target", "_prediction"), validate="one_to_one"
    )
    if len(merged) != len(targets) or len(merged) != len(oof):
        raise ValueError("OOF target and prediction IDs must match exactly before calibration")
    target_array = merged[[f"{label}_target" for label in labels]].to_numpy()
    oof_array = merged[[f"{label}_prediction" for label in labels]].to_numpy()
    calibrated = test.copy()
    calibrated.loc[:, labels] = TemperatureScaler().fit(oof_array, target_array).transform(test[labels].to_numpy())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(output, index=False, float_format="%.7f")


if __name__ == "__main__":
    main()
