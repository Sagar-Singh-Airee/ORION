"""Health-check and cache raw DICOM studies before expensive training.

Usage:
    python preprocess.py --config config.yaml --data-dir /data/knee --output manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.dicom.reader import load_study  # noqa: E402
from orion.utils.config import load_config  # noqa: E402

__all__ = ["main", "resolve_studies", "health_check_studies"]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Only check the first N studies (0 = all)")
    parser.add_argument("--output", default="preprocess_manifest.json")
    parser.add_argument("--progress-every", type=int, default=200, help="Print progress every N studies; 0 disables")
    args, overrides = parser.parse_known_args()
    if args.limit < 0:
        parser.error(f"--limit must be >= 0, got {args.limit}")
    return args, overrides


def resolve_studies(root: Path, limit: int) -> list[Path]:
    """List study directories in a fixed, sorted order so --limit is reproducible across runs."""
    if not root.exists():
        raise FileNotFoundError(f"data root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"data root is not a directory: {root}")
    studies = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
    if not studies:
        print(f"Warning: no study subdirectories found under {root}")
    return studies[:limit] if limit else studies


def health_check_studies(studies: list[Path], progress_every: int) -> dict:
    """Load every study, recording per-study outcomes without letting one bad study stop the rest."""
    valid_studies = 0
    total_series = 0
    series_counts: list[int] = []
    failures: list[dict] = []
    error_type_counts: dict[str, int] = {}
    total = len(studies)

    for index, study in enumerate(studies):
        try:
            series = load_study(study)
        except Exception as exc:  # noqa: BLE001 - record every bad study rather than stop after one
            error_type = type(exc).__name__
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            failures.append({"study": study.name, "error_type": error_type, "error": str(exc)})
        else:
            n_series = len(series) if series else 0
            total_series += n_series
            if n_series:
                valid_studies += 1
                series_counts.append(n_series)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"Checked {index + 1}/{total} studies...")

    series_stats = None
    if series_counts:
        ordered = sorted(series_counts)
        series_stats = {"min": ordered[0], "median": ordered[len(ordered) // 2], "max": ordered[-1]}

    return {
        "studies": total,
        "valid_studies": valid_studies,
        "series": total_series,
        "series_per_valid_study": series_stats,
        "error_type_counts": error_type_counts,
        "failures": failures,
    }


def main() -> None:
    args, overrides = parse_args()
    cfg = load_config(args.config, [f"data.root={args.data_dir}", *overrides])
    root = Path(cfg.data.root)

    studies = resolve_studies(root, args.limit)
    result = health_check_studies(studies, args.progress_every)
    report = {"root": str(root), **result}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Full failure list goes to the report file; the console gets a bounded preview
    # so a broken data path doesn't dump thousands of near-identical lines.
    console_report = dict(report)
    if len(console_report["failures"]) > 10:
        remaining = len(report["failures"]) - 10
        console_report["failures"] = report["failures"][:10] + [{"note": f"... {remaining} more failure(s); see {output}"}]
    print(json.dumps(console_report, indent=2))

    if studies and report["valid_studies"] == 0:
        print(f"ERROR: 0 of {len(studies)} studies passed the health check; see {output} for details", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()