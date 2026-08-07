"""Config loading and merging.

Resolution order (later wins): default.yaml -> configs/{data,model,training,inference}/*.yaml
referenced by the experiment file -> experiment/<name>.yaml -> CLI dotlist overrides.

Usage:
    cfg = load_config("experiment/baseline_resnet50.yaml", overrides=["training.optimizer.lr=3e-4"])
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def _load_yaml(rel_path: str | Path) -> DictConfig:
    path = CONFIG_ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path)


def load_config(experiment_path: str, overrides: list[str] | None = None) -> DictConfig:
    """Build a fully-merged config for one experiment.

    `experiment_path` is relative to configs/, e.g. "experiment/baseline_resnet50.yaml".
    The experiment file may declare an `includes:` list of other config paths (relative
    to configs/) that are merged in order before the experiment file's own keys are applied.
    """
    base = _load_yaml("default.yaml")
    exp = _load_yaml(experiment_path)

    merged = base
    for inc in exp.get("includes", []):
        merged = OmegaConf.merge(merged, _load_yaml(inc))

    exp_body = OmegaConf.create({k: v for k, v in exp.items() if k != "includes"})
    merged = OmegaConf.merge(merged, exp_body)

    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    OmegaConf.set_readonly(merged, False)
    return merged


def save_config(cfg: DictConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def to_container(cfg: DictConfig) -> dict:
    """Plain dict, resolved — safe for JSON/W&B logging."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]