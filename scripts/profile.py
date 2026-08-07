"""Profile DICOM discovery/load cost on a bounded number of studies."""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.dicom.reader import load_study


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", required=True); parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(); studies = [item for item in Path(args.data_dir).iterdir() if item.is_dir()][:args.limit]; timings=[]
    for study in studies:
        start=perf_counter(); load_study(study); timings.append(perf_counter()-start)
    print({"studies":len(timings), "mean_seconds":statistics.mean(timings) if timings else 0, "p95_seconds":sorted(timings)[max(0, int(.95*len(timings))-1)] if timings else 0})


if __name__ == "__main__": main()
