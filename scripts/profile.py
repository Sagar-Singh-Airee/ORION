"""Profile DICOM discovery/load cost on a bounded number of studies.

Usage:
    python profile.py --data-dir /data/knee --limit 50 --warmup 5 --output profile.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.dicom.reader import load_study  # noqa: E402

__all__ = ["main", "profile_studies", "compute_timing_stats"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--limit", type=int, default=20, help="Number of studies to profile")
    parser.add_argument("--warmup", type=int, default=0, help="Discard the first N successful loads from the stats (cold-cache/JIT effects)")
    parser.add_argument("--output", help="Optional path to also write the profile as JSON")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N studies; 0 disables")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error(f"--limit must be >= 1, got {args.limit}")
    if args.warmup < 0:
        parser.error(f"--warmup must be >= 0, got {args.warmup}")
    return args


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; well-defined even for very small samples."""
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    index = max(0, int(fraction * len(sorted_values)) - 1)
    return sorted_values[min(index, len(sorted_values) - 1)]


def profile_studies(studies: list[Path], progress_every: int) -> tuple[list[float], list[dict]]:
    """Time load_study per study, tolerating failures so one bad study doesn't lose all timings."""
    timings: list[float] = []
    failures: list[dict] = []
    total = len(studies)
    for index, study in enumerate(studies):
        start = perf_counter()
        try:
            load_study(study)
        except Exception as exc:  # noqa: BLE001 - keep profiling the rest even if one study is broken
            failures.append({"study": study.name, "error_type": type(exc).__name__, "error": str(exc)})
        else:
            timings.append(perf_counter() - start)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"Profiled {index + 1}/{total} studies...")
    return timings, failures


def compute_timing_stats(timings: list[float], warmup: int) -> dict:
    measured = timings[warmup:] if warmup else timings
    if not measured:
        return {
            "n": 0, "mean_seconds": None, "median_seconds": None, "p95_seconds": None,
            "min_seconds": None, "max_seconds": None, "total_seconds": 0.0,
            "throughput_studies_per_second": None,
        }
    ordered = sorted(measured)
    total_seconds = sum(measured)
    return {
        "n": len(measured),
        "mean_seconds": round(statistics.mean(measured), 4),
        "median_seconds": round(statistics.median(measured), 4),
        "p95_seconds": round(_percentile(ordered, 0.95), 4),
        "min_seconds": round(ordered[0], 4),
        "max_seconds": round(ordered[-1], 4),
        "total_seconds": round(total_seconds, 4),
        "throughput_studies_per_second": round(len(measured) / total_seconds, 4) if total_seconds > 0 else None,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data dir is not a directory: {data_dir}")

    # Sorted so `--limit` profiles the same studies on every run, making results comparable.
    studies = sorted((path for path in data_dir.iterdir() if path.is_dir()), key=lambda path: path.name)[: args.limit]
    if not studies:
        raise ValueError(f"No study subdirectories found under {data_dir} (or --limit too small)")

    timings, failures = profile_studies(studies, args.progress_every)
    if args.warmup and args.warmup >= len(timings):
        print(f"Warning: --warmup {args.warmup} >= {len(timings)} successful load(s); no timings left after warmup")

    report = {
        "data_dir": str(data_dir),
        "requested_studies": len(studies),
        "successful_loads": len(timings),
        "failed_loads": len(failures),
        "warmup_discarded": min(args.warmup, len(timings)),
        "timing": compute_timing_stats(timings, args.warmup),
        "failures": failures[:10] + ([{"note": f"... {len(failures) - 10} more"}] if len(failures) > 10 else []),
    }
    rendered = json.dumps(report, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")

    print(rendered)


if __name__ == "__main__":
    main()