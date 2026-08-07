"""
File I/O and Checkpoint Utilities

WHY it exists:
Handles reading/writing data formats safely, parsing CSVs, saving model checkpoints,
and tracking best metrics for model saving.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import torch
from loguru import logger

def save_checkpoint(state: Dict[str, Any], is_best: bool, save_dir: str | Path, filename: str = "checkpoint.pth") -> None:
    """
    Saves a PyTorch model checkpoint.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / filename
    
    torch.save(state, file_path)
    logger.debug(f"Saved checkpoint to {file_path}")
    
    if is_best:
        best_path = save_dir / "best_model.pth"
        torch.save(state, best_path)
        logger.info(f"New best model saved to {best_path}")

def load_checkpoint(checkpoint_path: str | Path, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    """
    Loads a PyTorch checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint # type: ignore

def save_json(data: Any, file_path: str | Path) -> None:
    """Saves data to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(file_path: str | Path) -> Any:
    """Loads data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def save_pickle(data: Any, file_path: str | Path) -> None:
    """Saves data to a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(file_path: str | Path) -> Any:
    """Loads data from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)
