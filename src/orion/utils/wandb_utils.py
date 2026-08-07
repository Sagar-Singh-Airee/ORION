"""
Weights & Biases (W&B) Logging Utility

WHY it exists:
Tracking experiments (loss curves, metrics, hyperparameters, images) is critical.
W&B provides a centralized dashboard. This utility manages initialization
and ensures we don't crash if W&B is disabled or offline.
"""

import os
from typing import Dict, Any, Optional
from loguru import logger
from omegaconf import DictConfig, OmegaConf

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def init_wandb(config: DictConfig) -> None:
    """
    Initializes W&B run if enabled in config.
    """
    wandb_cfg = config.get("logging", {}).get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        logger.info("W&B logging is disabled.")
        return

    if not WANDB_AVAILABLE:
        logger.warning("W&B enabled in config but wandb package not installed.")
        return

    project = wandb_cfg.get("project", "orion")
    entity = wandb_cfg.get("entity", None)
    
    # Flatten config for W&B
    flat_config = OmegaConf.to_container(config, resolve=True)
    
    run_name = config.get("experiment", {}).get("name", "unnamed_run")
    tags = config.get("experiment", {}).get("tags", [])
    notes = config.get("experiment", {}).get("notes", "")

    try:
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=flat_config, # type: ignore
            tags=tags,
            notes=notes,
            resume="allow"
        )
        logger.info(f"Initialized W&B run: {run_name}")
    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Logs metrics to W&B if active.
    """
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb() -> None:
    """
    Closes the W&B run.
    """
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        logger.info("W&B run finished.")
