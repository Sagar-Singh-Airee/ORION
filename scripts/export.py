"""Export a trained ORION checkpoint to ONNX after model training is complete."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.inference.onnx_export import export_onnx
from orion.models.architectures import ORIONMultimodalModel
from orion.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); cfg = load_config(args.config); model = ORIONMultimodalModel(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False); model.load_state_dict(state.get("model", state.get("state_dict", state)))
    print(export_onnx(model, args.output, (int(cfg.data.image_size), int(cfg.data.image_size)), int(cfg.data.num_slices), int(cfg.labels.num_classes)))


if __name__ == "__main__": main()
