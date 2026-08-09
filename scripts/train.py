"""Train one leakage-safe fold in Kaggle; no training is executed by this repository itself.

Usage:
    python train.py --config config.yaml training.max_epochs=30 optimizer.lr=3e-4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.datasets import KneeMRIDataset, MultimodalKneeDataset, variable_length_collate  # noqa: E402
from orion.data.text.label_extractor import FINDINGS  # noqa: E402
from orion.data.transforms import create_train_transforms, create_val_transforms  # noqa: E402
from orion.models.architectures import ORIONMultimodalModel  # noqa: E402
from orion.models.ema import ModelEMA  # noqa: E402
from orion.models.losses import AsymmetricLoss, FocalLoss, MaskedBCEWithLogitsLoss  # noqa: E402
from orion.training.optimizer import create_adamw, create_sgd  # noqa: E402
from orion.training.reproducibility import seed_everything, seeded_generator, worker_init_fn  # noqa: E402
from orion.training.scheduler import CosineAnnealingWithWarmup  # noqa: E402
from orion.training.trainer import Trainer  # noqa: E402
from orion.utils.config import load_config, save_config  # noqa: E402
from orion.utils.logging import setup_logger  # noqa: E402

__all__ = ["main", "build_loss", "build_optimizer", "build_ema", "resolve_label_names"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Allow training into a non-empty output directory")
    parser.add_argument("overrides", nargs="*", help="Config overrides in key=value form, e.g. training.max_epochs=30")
    return parser.parse_args()


def validate_overrides(overrides: list[str]) -> None:
    malformed = [item for item in overrides if "=" not in item]
    if malformed:
        raise ValueError(f"Overrides must be in key=value form; got malformed token(s): {malformed}")


def resolve_output_dir(cfg) -> Path:
    experiment_cfg = cfg.get("experiment", {}) if hasattr(cfg, "get") else {}
    experiment_name = experiment_cfg.get("name", "orion") if hasattr(experiment_cfg, "get") else "orion"
    return Path(cfg.paths.output_dir) / experiment_name


def prepare_output_dir(output: Path, overwrite: bool) -> None:
    """Refuse to silently train into (and overwrite) a previous run's directory."""
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output} already exists and is not empty. Pass --overwrite to train into it anyway, "
            "or change experiment.name / paths.output_dir to avoid clobbering a previous run."
        )
    output.mkdir(parents=True, exist_ok=True)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA is not available; training will run on CPU and will be dramatically slower.")
    return device


def build_datasets(cfg) -> tuple:
    dataset_type = MultimodalKneeDataset if cfg.data.get("multimodal", False) else KneeMRIDataset
    train_dataset = dataset_type(cfg, "train", transform=create_train_transforms(cfg))
    val_dataset = dataset_type(cfg, "val", transform=create_val_transforms(cfg))
    if len(train_dataset) == 0:
        raise ValueError("Train dataset resolved 0 rows; check cfg.data settings")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset resolved 0 rows; check cfg.data settings")
    return train_dataset, val_dataset


def build_loaders(cfg, train_dataset, val_dataset) -> tuple[DataLoader, DataLoader]:
    loader_kwargs = {
        "batch_size": int(cfg.training.batch_size),
        "num_workers": int(cfg.data.num_workers),
        "pin_memory": bool(cfg.data.pin_memory),
        "collate_fn": variable_length_collate,
        "worker_init_fn": worker_init_fn,
        "persistent_workers": bool(cfg.data.persistent_workers and cfg.data.num_workers > 0),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=seeded_generator(int(cfg.project.seed)), **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def build_loss(cfg):
    loss_cfg = cfg.loss.primary
    name = loss_cfg.name
    if name == "bce":
        return MaskedBCEWithLogitsLoss(loss_cfg.get("pos_weight"))
    if name == "focal":
        return FocalLoss(loss_cfg.get("gamma", 2.0), loss_cfg.get("alpha"))
    if name == "asymmetric":
        return AsymmetricLoss(loss_cfg.get("gamma_neg", 4.0), loss_cfg.get("gamma_pos", 0.0), loss_cfg.get("clip", 0.05))
    raise ValueError(f"Unknown loss {name!r}; expected 'bce', 'focal', or 'asymmetric'")


def build_optimizer(model, optimizer_cfg):
    """Dispatch by name, and fail loudly on an unrecognized one.

    The original silently built SGD for *any* name other than 'adamw' — a typo like
    'adam' or a not-yet-wired name like 'lamb' would train with the wrong optimizer
    and no error, which is a hard bug to notice from results alone.
    """
    name = optimizer_cfg.name
    if name == "adamw":
        return create_adamw(model, optimizer_cfg)
    if name == "sgd":
        return create_sgd(model, optimizer_cfg)
    raise ValueError(f"Unknown optimizer {name!r}; expected 'adamw' or 'sgd'")


def resolve_ema_config(cfg):
    """EMA settings may live under cfg.model.ema or cfg.training.ema; check both explicitly
    instead of silently preferring one location when both happen to be present.
    """
    model_ema = cfg.model.get("ema", {}) if hasattr(cfg.model, "get") else {}
    training_ema = cfg.training.get("ema", {}) if hasattr(cfg.training, "get") else {}
    if model_ema.get("enabled", False) and training_ema.get("enabled", False):
        print("Note: EMA config present in both cfg.model.ema and cfg.training.ema; using cfg.model.ema")
    return model_ema if model_ema else training_ema


def build_ema(model, cfg, device) -> ModelEMA | None:
    ema_cfg = resolve_ema_config(cfg)
    if not ema_cfg.get("enabled", False):
        return None
    decay = ema_cfg.get("decay")
    if decay is None:
        raise ValueError("EMA is enabled but no 'decay' value was found in cfg.model.ema / cfg.training.ema")
    return ModelEMA(model, float(decay), device)


def resolve_label_names(cfg) -> list[str]:
    labels_cfg = getattr(cfg, "labels", None)
    names = None
    if labels_cfg is not None:
        names = labels_cfg.get("names") if hasattr(labels_cfg, "get") else getattr(labels_cfg, "names", None)
    return list(names) if names else list(FINDINGS)


def main() -> None:
    args = parse_args()
    validate_overrides(args.overrides)
    cfg = load_config(args.config, args.overrides)

    output = resolve_output_dir(cfg)
    prepare_output_dir(output, args.overwrite)
    save_config(cfg, output / "config.yaml")
    setup_logger(cfg, output)

    seed_everything(int(cfg.project.seed), bool(cfg.project.deterministic))
    device = resolve_device()

    train_dataset, val_dataset = build_datasets(cfg)
    train_loader, val_loader = build_loaders(cfg, train_dataset, val_dataset)

    # Move the model to its target device before building the optimizer/EMA, rather than
    # relying on Trainer to do it internally at an unknown point.
    model = ORIONMultimodalModel(cfg).to(device)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = CosineAnnealingWithWarmup(
        optimizer, int(cfg.scheduler.warmup_epochs), int(cfg.training.max_epochs), float(cfg.scheduler.min_lr)
    )
    ema = build_ema(model, cfg, device)
    labels = resolve_label_names(cfg)

    print(f"Training {len(train_dataset)} train / {len(val_dataset)} val sample(s), {len(labels)} label(s) -> {output}")

    trainer = Trainer(model, optimizer, build_loss(cfg), device, train_loader, val_loader, cfg, scheduler, ema, labels)
    history = trainer.fit(output)

    (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=True, default=str), encoding="utf-8")

    if isinstance(history, dict):
        print(f"Training complete. Recorded {len(history)} metric key(s). Artifacts -> {output}")
    elif isinstance(history, list):
        print(f"Training complete. {len(history)} epoch(s) recorded. Artifacts -> {output}")
    else:
        print(f"Training complete. Artifacts -> {output}")


if __name__ == "__main__":
    main()