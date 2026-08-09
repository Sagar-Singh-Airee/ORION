"""Create ternary weak labels and an evidence audit trail from report text.

Usage:
    python extract_weak_labels.py --reports-csv reports.csv --output weak_labels.csv \\
        --id-column study_id --evidence-output evidence.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.text.label_extractor import FINDINGS, extract_labels  # noqa: E402

__all__ = ["main", "extract_all", "assign_labels", "validate_ternary_labels", "compute_finding_summary"]

_ALLOWED_LABEL_VALUES = {-1, 0, 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="report_text")
    parser.add_argument("--id-column", help="Optional id column, included in the evidence trail so it stays joinable if rows are later reordered")
    parser.add_argument("--evidence-output", help="Optional JSONL evidence path")
    parser.add_argument("--progress-every", type=int, default=2000, help="Print progress every N reports; 0 disables")
    return parser.parse_args()


def extract_all(frame: pd.DataFrame, text_column: str, progress_every: int) -> tuple[list, list[int], int]:
    """Run extract_labels per report, tolerating per-row failures and missing text.

    A single malformed report must not abort extraction for the other thousands of rows.
    Returns (results, failed_row_indices, n_missing_text); failed rows get an empty
    placeholder result so downstream assembly needs no special-casing, but the failure
    is still reported rather than silently swallowed.
    """
    results = []
    failed_rows: list[int] = []
    n_missing_text = 0
    total = len(frame)
    for index, text in enumerate(frame[text_column]):
        if pd.isna(text):
            n_missing_text += 1
            text_value = ""
        else:
            text_value = str(text)
        try:
            result = extract_labels(text_value)
        except Exception as exc:  # noqa: BLE001 - one bad report must not abort the whole run
            failed_rows.append(index)
            result = types.SimpleNamespace(labels={}, evidence=[])
            print(f"Warning: extraction failed for row {index}: {exc}")
        results.append(result)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"Processed {index + 1}/{total} reports...")
    return results, failed_rows, n_missing_text


def assign_labels(frame: pd.DataFrame, findings: list[str], results: list) -> None:
    for finding in findings:
        if finding in frame.columns:
            print(f"Note: overwriting existing column {finding!r} with extracted weak labels")
        frame[finding] = [result.labels.get(finding, -1) for result in results]


def validate_ternary_labels(frame: pd.DataFrame, findings: list[str]) -> None:
    """Catch an extractor contract violation immediately, not three pipeline stages later."""
    problems = {}
    for finding in findings:
        bad_values = set(frame[finding].unique()) - _ALLOWED_LABEL_VALUES
        if bad_values:
            problems[finding] = sorted(bad_values)
    if problems:
        raise ValueError(f"extract_labels produced values outside {_ALLOWED_LABEL_VALUES} for: {problems}")


def write_evidence(path: Path, results: list, id_values: list | None) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, result in enumerate(results):
            entry = {"row": index, "evidence": result.evidence}
            if id_values is not None:
                entry["id"] = id_values[index]
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def compute_finding_summary(frame: pd.DataFrame, findings: list[str]) -> dict:
    """Coverage (fraction with a known label) and positive rate among known labels per finding.

    Coverage alone hides the more decision-relevant number: among reports where a finding
    was actually mentioned, how often is it positive. Both matter for weighting the loss.
    """
    summary = {}
    for finding in findings:
        column = frame[finding]
        known = column[column >= 0]
        summary[finding] = {
            "coverage": float((column >= 0).mean()),
            "positive_rate": float((known == 1).mean()) if len(known) else None,
        }
    return summary


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.reports_csv)
    if frame.empty:
        raise ValueError(f"{args.reports_csv} contains no rows")
    if args.text_column not in frame.columns:
        raise ValueError(f"{args.text_column!r} not in {list(frame.columns)}")
    if args.id_column and args.id_column not in frame.columns:
        raise ValueError(f"--id-column {args.id_column!r} not in {list(frame.columns)}")

    results, failed_rows, n_missing_text = extract_all(frame, args.text_column, args.progress_every)
    assign_labels(frame, FINDINGS, results)
    validate_ternary_labels(frame, FINDINGS)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    if args.evidence_output:
        evidence_path = Path(args.evidence_output)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        id_values = frame[args.id_column].tolist() if args.id_column else None
        write_evidence(evidence_path, results, id_values)

    summary = {
        "rows": len(frame),
        "output": str(output),
        "n_missing_text": n_missing_text,
        "n_extraction_errors": len(failed_rows),
        "failed_rows_sample": failed_rows[:10],
        "coverage": compute_finding_summary(frame, FINDINGS),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()