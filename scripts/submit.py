"""Validate an existing prediction CSV and write Kaggle-ready submission CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.inference.submission import create_submission, save_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="study_id")
    parser.add_argument("--sample-submission", help="Use its exact label order")
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    if args.id_column not in frame: raise ValueError(f"Missing ID column {args.id_column!r}")
    labels = [column for column in pd.read_csv(args.sample_submission).columns if column != args.id_column] if args.sample_submission else [column for column in frame.columns if column != args.id_column]
    result = create_submission(frame[args.id_column], frame[labels].to_numpy(), labels)
    result = result.rename(columns={"study_id": args.id_column}) if args.id_column != "study_id" else result
    save_submission(result, args.output)


if __name__ == "__main__": main()
