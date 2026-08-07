"""Ensemble aligned OOF/test prediction CSVs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.text.label_extractor import FINDINGS
from orion.inference.ensemble import ensemble_predictions
from orion.inference.submission import save_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction CSV files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="study_id")
    parser.add_argument("--method", choices=("mean", "median", "rank_mean"), default="mean")
    parser.add_argument("--weights", type=float, nargs="*")
    args = parser.parse_args()
    frames = [pd.read_csv(path) for path in args.predictions]
    labels = [label for label in FINDINGS if all(label in frame for frame in frames)]
    if not labels: raise ValueError("Prediction files must share ORION label columns")
    ids = frames[0][args.id_column].astype(str)
    for frame in frames[1:]:
        if not ids.equals(frame[args.id_column].astype(str)): raise ValueError("Prediction files have different ID order")
    result = pd.DataFrame(ensemble_predictions([frame[labels].to_numpy() for frame in frames], args.weights, args.method), columns=labels)
    result.insert(0, args.id_column, ids)
    save_submission(result, args.output)


if __name__ == "__main__": main()
