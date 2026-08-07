"""Train one leakage-safe fold in Kaggle; no training is executed by this repository itself."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orion.data.datasets import KneeMRIDataset, MultimodalKneeDataset, variable_length_collate
from orion.data.text.label_extractor import FINDINGS
from orion.data.transforms import create_train_transforms, create_val_transforms
from orion.models.architectures import ORIONMultimodalModel
from orion.models.ema import ModelEMA
from orion.models.losses import AsymmetricLoss, FocalLoss, MaskedBCEWithLogitsLoss
from orion.training.optimizer import create_adamw, create_sgd
from orion.training.reproducibility import seed_everything, seeded_generator, worker_init_fn
from orion.training.scheduler import CosineAnnealingWithWarmup
from orion.training.trainer import Trainer
from orion.utils.config import load_config, save_config
from orion.utils.logging import setup_logger


def build_loss(config):
    loss = config.loss.primary; name = loss.name
    if name == "bce": return MaskedBCEWithLogitsLoss(loss.get("pos_weight"))
    if name == "focal": return FocalLoss(loss.get("gamma", 2.0), loss.get("alpha"))
    if name == "asymmetric": return AsymmetricLoss(loss.get("gamma_neg", 4.0), loss.get("gamma_pos", 0.0), loss.get("clip", 0.05))
    raise ValueError(f"Unknown loss {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(); cfg = load_config(args.config, args.overrides)
    output = Path(cfg.paths.output_dir) / cfg.get("experiment", {}).get("name", "orion")
    output.mkdir(parents=True, exist_ok=True); save_config(cfg, output / "config.yaml"); setup_logger(cfg, output)
    seed_everything(int(cfg.project.seed), bool(cfg.project.deterministic)); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_type = MultimodalKneeDataset if cfg.data.get("multimodal", False) else KneeMRIDataset
    train = dataset_type(cfg, "train", transform=create_train_transforms(cfg)); validation = dataset_type(cfg, "val", transform=create_val_transforms(cfg))
    loader_kwargs = {"batch_size": int(cfg.training.batch_size), "num_workers": int(cfg.data.num_workers), "pin_memory": bool(cfg.data.pin_memory), "collate_fn": variable_length_collate, "worker_init_fn": worker_init_fn, "persistent_workers": bool(cfg.data.persistent_workers and cfg.data.num_workers > 0)}
    train_loader = DataLoader(train, shuffle=True, generator=seeded_generator(int(cfg.project.seed)), **loader_kwargs); val_loader = DataLoader(validation, shuffle=False, **loader_kwargs)
    model = ORIONMultimodalModel(cfg); optimizer = create_adamw(model, cfg.optimizer) if cfg.optimizer.name == "adamw" else create_sgd(model, cfg.optimizer)
    scheduler = CosineAnnealingWithWarmup(optimizer, int(cfg.scheduler.warmup_epochs), int(cfg.training.max_epochs), float(cfg.scheduler.min_lr))
    ema_cfg = cfg.model.get("ema", cfg.training.get("ema", {}))
    ema = ModelEMA(model, float(ema_cfg.decay), device) if ema_cfg.get("enabled", False) else None
    history = Trainer(model, optimizer, build_loss(cfg), device, train_loader, val_loader, cfg, scheduler, ema, list(cfg.labels.names or FINDINGS)).fit(output)
    (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__": main()
