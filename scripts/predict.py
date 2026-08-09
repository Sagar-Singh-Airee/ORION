"""Run a trained checkpoint against a test CSV and write validated probabilities.

Usage:
    python predict.py --config config.yaml --checkpoint best.pt --output submission.csv \\
        --data-root /data/knee --metadata-csv test.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.datasets import KneeMRIDataset, MultimodalKneeDataset, variable_length_collate  # noqa: E402
from orion.data.text.label_extractor import FINDINGS  # noqa: E402
from orion.inference import Predictor, create_submission, save_submission  # noqa: E402
from orion.models.architectures import ORIONMultimodalModel  # noqa: E402
from orion.utils.config import load_config  # noqa: E402

__all__ = ["main", "load_checkpoint", "resolve_state_dict", "resolve_tta_strategies", "resolve_label_names"]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--metadata-csv")
    args, overrides = parser.parse_known_args()
    if args.data_root:
        overrides.append(f"data.root={args.data_root}")
    if args.metadata_csv:
        overrides.append(f"data.test_csv={args.metadata_csv}")
    return args, overrides


def load_checkpoint(checkpoint_path: str) -> Any:
    """Prefer safe (weights_only) loading; only fall back for legacy checkpoints, and say so."""
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - genuinely need to catch and retry here
        print(
            f"Note: safe checkpoint load (weights_only=True) failed ({exc}); "
            "retrying with weights_only=False. Only do this for checkpoints you trust."
        )
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def resolve_state_dict(checkpoint: Any, checkpoint_path: str) -> tuple[dict, str]:
    """Locate the actual state dict inside a checkpoint, reporting which key was used."""
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint at {checkpoint_path} is not a dict (got {type(checkpoint).__name__}); "
            "expected torch.save output with a 'model'/'state_dict' key, or a raw state dict."
        )
    for key in ("model", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate, key
    return checkpoint, "<checkpoint root>"


def resolve_tta_strategies(cfg: Any) -> list[str]:
    """Pull the TTA strategy list out of config, tolerating either a bare list or a
    {'strategies': [...]} block, and reject shapes that would silently misbehave
    (e.g. a bare string, which naive `list()` would split into characters).
    """
    inference_cfg = cfg.get("inference", {}) if hasattr(cfg, "get") else {}
    tta_cfg = inference_cfg.get("tta", ["identity"]) if hasattr(inference_cfg, "get") else ["identity"]
    strategies = tta_cfg.get("strategies", ["identity"]) if hasattr(tta_cfg, "get") else tta_cfg

    if isinstance(strategies, str):
        raise ValueError(f"cfg.inference.tta must be a list of strategy names, got a bare string {strategies!r}")
    strategies = list(strategies)
    if not strategies:
        raise ValueError("Resolved TTA strategy list is empty; expected at least ['identity']")
    return strategies


def resolve_label_names(cfg: Any) -> list[str]:
    labels_cfg = getattr(cfg, "labels", None)
    names = None
    if labels_cfg is not None:
        names = labels_cfg.get("names") if hasattr(labels_cfg, "get") else getattr(labels_cfg, "names", None)
    return list(names) if names else list(FINDINGS)


def build_dataloader(cfg: Any) -> DataLoader:
    dataset_type = MultimodalKneeDataset if cfg.data.get("multimodal", False) else KneeMRIDataset
    dataset = dataset_type(cfg, split="test")
    if len(dataset) == 0:
        raise ValueError("Test dataset resolved 0 rows; check cfg.data.root / cfg.data.test_csv")
    return DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        collate_fn=variable_length_collate,
    )


def main() -> None:
    args, overrides = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = load_config(args.config, overrides)
    loader = build_dataloader(cfg)

    model = ORIONMultimodalModel(cfg)
    checkpoint = load_checkpoint(str(checkpoint_path))
    state_dict, source_key = resolve_state_dict(checkpoint, str(checkpoint_path))
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load state dict (source key: {source_key!r}) from {checkpoint_path}: {exc}"
        ) from exc

    # Critical: without eval(), dropout stays active and batchnorm uses batch statistics
    # instead of running statistics, silently corrupting every prediction.
    model.eval()

    tta = resolve_tta_strategies(cfg)
    labels = resolve_label_names(cfg)

    with torch.no_grad():
        ids, probabilities = Predictor(model, tta=tta).predict_loader(loader)

    if len(ids) != len(probabilities):
        raise ValueError(f"Predictor returned {len(ids)} id(s) but {len(probabilities)} row(s) of probabilities")
    if probabilities.shape[1] != len(labels):
        raise ValueError(
            f"Model produced {probabilities.shape[1]} label column(s) but {len(labels)} label name(s) "
            f"were resolved ({labels}); check cfg.labels.names"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_submission(create_submission(ids, probabilities, labels), args.output)

    print(f"Loaded weights from {checkpoint_path} (source key: {source_key!r})")
    print(f"Predicted {len(ids)} row(s) over {len(labels)} label(s) with TTA={tta} -> {output_path}")


if __name__ == "__main__":
    main()