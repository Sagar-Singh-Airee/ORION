"""Health-check and cache raw DICOM studies before expensive training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.dicom.reader import load_study
from orion.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="preprocess_manifest.json")
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, [f"data.root={args.data_dir}", *overrides])
    root = Path(cfg.data.root)
    studies = [path for path in root.iterdir() if path.is_dir()]
    if args.limit: studies = studies[:args.limit]
    report = {"root": str(root), "studies": len(studies), "valid_studies": 0, "series": 0, "failures": []}
    for study in studies:
        try:
            series = load_study(study)
            if series: report["valid_studies"] += 1
            report["series"] += len(series)
        except Exception as exc:  # record every bad study rather than stop after one
            report["failures"].append({"study": study.name, "error": str(exc)})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
