"""Evaluate OOF predictions; all metric inputs are aligned by study ID."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.text.label_extractor import FINDINGS
from orion.evaluation.metrics import calculate_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--id-column", default="study_id")
    parser.add_argument("--output")
    args = parser.parse_args()
    targets, predictions = pd.read_csv(args.targets), pd.read_csv(args.predictions)
    labels = [label for label in FINDINGS if label in targets and label in predictions]
    if not labels: raise ValueError("No shared target columns; expected canonical ORION labels")
    if args.id_column in targets and args.id_column in predictions:
        merged = targets[[args.id_column, *labels]].merge(predictions[[args.id_column, *labels]], on=args.id_column, suffixes=("_target", "_prediction"), validate="one_to_one")
        truth = merged[[f"{label}_target" for label in labels]].to_numpy()
        probability = merged[[f"{label}_prediction" for label in labels]].to_numpy()
    else:
        if len(targets) != len(predictions): raise ValueError("Rows differ and no shared ID column exists")
        truth, probability = targets[labels].to_numpy(), predictions[labels].to_numpy()
    metrics = calculate_metrics(truth, probability, labels)
    rendered = json.dumps(metrics, indent=2, allow_nan=True)
    if args.output: Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__": main()
