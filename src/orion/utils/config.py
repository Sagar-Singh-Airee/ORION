"""
Configuration Management Utility

WHY it exists:
Hardcoding hyperparameters in scripts leads to untrackable experiments.
We use Hydra/OmegaConf to manage complex hierarchical configurations.
This allows us to compose configs (e.g., base + model + data) and override
values from the command line without changing code.

Implementation:
Uses `omegaconf` for parsing and merging YAML files.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str | Path, cli_args: Optional[list[str]] = None) -> DictConfig:
    """
    Loads a YAML configuration file, merges defaults, and applies CLI overrides.
    
    Args:
        config_path: Path to the main YAML configuration file.
        cli_args: List of command-line arguments (e.g., ["model.vision_backbone.lr=1e-4"]).
        
    Returns:
        OmegaConf DictConfig object containing the full merged configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load base configuration
    base_config = OmegaConf.load(config_path)
    
    # Handle 'defaults' list if present (similar to Hydra's compose)
    if "defaults" in base_config:
        merged_config = OmegaConf.create()
        for default_item in base_config.defaults:
            if isinstance(default_item, str):
                # Resolve default path relative to the current config directory
                default_path = config_path.parent / f"{default_item}.yaml"
                if default_path.exists():
                    default_cfg = OmegaConf.load(default_path)
                    merged_config = OmegaConf.merge(merged_config, default_cfg)
            elif isinstance(default_item, dict):
                 # Handle dict-based defaults (e.g. - data: base_data)
                 for key, val in default_item.items():
                     default_path = config_path.parent / key / f"{val}.yaml"
                     if default_path.exists():
                         default_cfg = OmegaConf.load(default_path)
                         merged_config = OmegaConf.merge(merged_config, default_cfg)
        
        # Merge the base config on top of the defaults
        config = OmegaConf.merge(merged_config, base_config)
        # Remove defaults key from final config
        del config["defaults"]
    else:
        config = base_config

    # Apply command-line overrides
    if cli_args:
        cli_config = OmegaConf.from_cli(cli_args)
        config = OmegaConf.merge(config, cli_config)

    # Resolve environment variables (e.g., ${oc.env:DATA_DIR})
    OmegaConf.resolve(config)

    return config


def save_config(config: DictConfig, save_path: str | Path) -> None:
    """
    Saves a DictConfig to a YAML file. Useful for logging the exact
    configuration used for an experiment.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        OmegaConf.save(config, f)


def dictconfig_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Converts a DictConfig to a standard Python dictionary."""
    return OmegaConf.to_container(config, resolve=True) # type: ignore
