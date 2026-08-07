"""Emit a compact, reproducible data audit before selecting an experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--metadata", required=True); parser.add_argument("--output")
    args = parser.parse_args(); frame = pd.read_csv(args.metadata)
    summary = {"rows": len(frame), "columns": list(frame.columns), "missing_fraction": frame.isna().mean().round(4).to_dict(), "unique_counts": frame.nunique(dropna=True).to_dict()}
    rendered = json.dumps(summary, indent=2, default=str)
    if args.output: Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__": main()
