"""Create and persist leakage-safe folds; this does not train a model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.text.label_extractor import FINDINGS
from orion.training.cross_validation import make_fold_assignments, save_fold_assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-column", default="patient_id")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.metadata)
    labels = [label for label in FINDINGS if label in frame]
    if not labels: raise ValueError("Metadata needs ORION label columns to stratify folds")
    assigned = make_fold_assignments(frame, args.group_column, labels, args.n_folds, args.seed)
    print(save_fold_assignments(assigned, args.output))


if __name__ == "__main__": main()
