"""Export a trained ORION checkpoint to ONNX after model training is complete.

Usage:
    python export.py --config config.yaml --checkpoint best.pt --output model.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.inference.onnx_export import export_onnx  # noqa: E402
from orion.models.architectures import ORIONMultimodalModel  # noqa: E402
from orion.utils.config import load_config  # noqa: E402

__all__ = ["main", "load_checkpoint", "resolve_state_dict", "resolve_export_dimensions"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str) -> Any:
    """Load a checkpoint, preferring the safe (weights_only) path and falling back only if needed.

    ``weights_only=False`` unpickles arbitrary Python objects, which is a code-execution
    risk for checkpoints you didn't produce yourself. Try safe loading first; only fall
    back for legacy checkpoints that actually need it, and say so out loud when it happens.
    """
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


def _require_positive_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"Config is missing required field {name}")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def resolve_export_dimensions(cfg: Any) -> tuple[tuple[int, int], int, int]:
    """Pull and validate the shapes export_onnx needs, failing on the specific missing field."""
    image_size = _require_positive_int(getattr(getattr(cfg, "data", None), "image_size", None), "cfg.data.image_size")
    num_slices = _require_positive_int(getattr(getattr(cfg, "data", None), "num_slices", None), "cfg.data.num_slices")
    num_classes = _require_positive_int(
        getattr(getattr(cfg, "labels", None), "num_classes", None), "cfg.labels.num_classes"
    )
    return (image_size, image_size), num_slices, num_classes


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = load_config(args.config)
    image_shape, num_slices, num_classes = resolve_export_dimensions(cfg)

    model = ORIONMultimodalModel(cfg)
    checkpoint = load_checkpoint(str(checkpoint_path))
    state_dict, source_key = resolve_state_dict(checkpoint, str(checkpoint_path))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load state dict (source key: {source_key!r}) from {checkpoint_path}: {exc}"
        ) from exc

    # Critical: without eval(), dropout stays active and batchnorm uses batch statistics
    # instead of running statistics, silently corrupting the exported graph.
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        result = export_onnx(model, str(output_path), image_shape, num_slices, num_classes)

    print(f"Loaded weights from {checkpoint_path} (source key: {source_key!r})")
    print(f"Exported ONNX model -> {result if result else output_path}")


if __name__ == "__main__":
    main()