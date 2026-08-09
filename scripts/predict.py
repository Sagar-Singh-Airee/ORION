"""Run a trained checkpoint against a test CSV and write validated probabilities."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.datasets import KneeMRIDataset, MultimodalKneeDataset, variable_length_collate
from orion.data.text.label_extractor import FINDINGS
from orion.inference import Predictor, create_submission, save_submission
from orion.models.architectures import ORIONMultimodalModel
from orion.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--metadata-csv")
    args, overrides = parser.parse_known_args()
    if args.data_root: overrides.append(f"data.root={args.data_root}")
    if args.metadata_csv: overrides.append(f"data.test_csv={args.metadata_csv}")
    cfg = load_config(args.config, overrides)
    model = ORIONMultimodalModel(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state.get("state_dict", state)), strict=True)
    dataset_type = MultimodalKneeDataset if cfg.data.get("multimodal", False) else KneeMRIDataset
    dataset = dataset_type(cfg, split="test")
    loader = DataLoader(dataset, batch_size=int(cfg.training.batch_size), num_workers=int(cfg.data.num_workers), pin_memory=bool(cfg.data.pin_memory), collate_fn=variable_length_collate)
    tta_config = cfg.get("inference", {}).get("tta", ["identity"])
    tta = tta_config.get("strategies", ["identity"]) if hasattr(tta_config, "get") else tta_config
    ids, probabilities = Predictor(model, tta=tta).predict_loader(loader)  # type: ignore[union-attr]
    save_submission(create_submission(ids, probabilities, list(cfg.labels.names or FINDINGS)), args.output)


if __name__ == "__main__": main()
