"""Create ternary weak labels and an evidence audit trail from report text."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.text.label_extractor import FINDINGS, extract_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="report_text")
    parser.add_argument("--evidence-output", help="Optional JSONL evidence path")
    args = parser.parse_args()
    frame = pd.read_csv(args.reports_csv)
    if args.text_column not in frame:
        raise ValueError(f"{args.text_column!r} not in {list(frame.columns)}")
    results = [extract_labels(str(text) if pd.notna(text) else "") for text in frame[args.text_column]]
    for finding in FINDINGS:
        frame[finding] = [result.labels.get(finding, -1) for result in results]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    if args.evidence_output:
        evidence_path = Path(args.evidence_output); evidence_path.parent.mkdir(parents=True, exist_ok=True)
        with evidence_path.open("w", encoding="utf-8") as handle:
            for index, result in enumerate(results):
                handle.write(json.dumps({"row": index, "evidence": result.evidence}, ensure_ascii=False) + "\n")
    coverage = {finding: float((frame[finding] >= 0).mean()) for finding in FINDINGS}
    print(json.dumps({"rows": len(frame), "output": str(output), "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
