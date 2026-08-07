"""
Training Script

Entry point for starting experiments.

Usage:
    python scripts/train.py --config configs/experiment/baseline_resnet50.yaml
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path so we can import orion
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from orion.utils.config import load_config
from orion.utils.logging import setup_logger
from orion.utils.wandb_utils import init_wandb, finish_wandb
from orion.utils.tensor_utils import get_device

# Mock imports for the script structure
from orion.data.datasets.knee_mri import KneeMRIDataset
from orion.models.architectures.multimodal import ORIONMultimodalModel
from orion.models.losses.asymmetric import AsymmetricLoss
from orion.training.optimizer import create_adamw, create_sgd
from orion.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="ORION Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("overrides", nargs="*", help="CLI overrides (e.g. training.lr=1e-3)")
    args = parser.parse_args()

    # 1. Load Config
    config = load_config(args.config, args.overrides)
    
    # 2. Setup Logging & Tracking
    setup_logger(config, config.paths.output_dir)
    init_wandb(config)
    
    device = get_device()
    logger.info(f"Using device: {device}")

    # 3. Data Loaders
    # In a real script, this uses the CrossValidation splits
    logger.info("Initializing Datasets...")
    train_dataset = KneeMRIDataset(config, split="train")
    # train_loader = DataLoader(train_dataset, batch_size=config.data.loader.batch_size, ...)
    
    # 4. Model
    logger.info(f"Initializing Model: {config.model.vision_backbone.name}")
    model = ORIONMultimodalModel(config).to(device)
    
    # 5. Optimizer & Loss
    optimizer = create_adamw(model, config.optimizer)
    criterion = AsymmetricLoss(gamma_neg=config.loss.primary.gamma_neg, clip=config.loss.primary.clip)
    
    # 6. Trainer
    # trainer = Trainer(...)
    # trainer.train_epoch(0)
    
    logger.info("Training script initialized successfully (Mock mode).")
    finish_wandb()

if __name__ == "__main__":
    main()
