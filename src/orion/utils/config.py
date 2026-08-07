"""Config loading and merging.

Resolution order (later wins): default.yaml -> configs/{data,model,training,inference}/*.yaml
referenced by the experiment file -> experiment/<name>.yaml -> CLI dotlist overrides.

Usage:
    cfg = load_config("experiment/baseline_resnet50.yaml", overrides=["training.optimizer.lr=3e-4"])
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def _resolve_path(path_like: str | Path, *, relative_to: Path | None = None) -> Path:
    """Resolve a config reference without depending on the caller's CWD."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    if relative_to is not None:
        candidate = (relative_to / path).resolve()
        if candidate.exists() or candidate.suffix:
            return candidate
    return (CONFIG_ROOT / path).resolve()


def _load_yaml(path_like: str | Path, *, relative_to: Path | None = None) -> tuple[DictConfig, Path]:
    path = _resolve_path(path_like, relative_to=relative_to)
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path), path


def _normalise_defaults(raw: object) -> Iterable[str]:
    """Support both Hydra-style ``defaults`` and this project's ``includes``."""
    if raw is None:
        return []
    refs: list[str] = []
    for item in raw:
        if isinstance(item, str):
            refs.append(item)
        elif isinstance(item, dict):
            refs.extend(str(value) for value in item.values() if value is not None)
    return refs


def load_config(experiment_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Build a fully-merged config for one experiment.

    `experiment_path` is relative to configs/, e.g. "experiment/baseline_resnet50.yaml".
    The experiment file may declare an `includes:` list of other config paths (relative
    to configs/) that are merged in order before the experiment file's own keys are applied.
    """
    base, _ = _load_yaml("default.yaml")
    exp, exp_path = _load_yaml(experiment_path)

    merged = base
    # Old experiment files use ``defaults: [../default]`` while newer files use
    # ``includes``. ``default.yaml`` is already loaded, so skipping a repeated
    # default keeps either spelling valid and avoids a circular merge.
    references = [*exp.get("includes", []), *_normalise_defaults(exp.get("defaults"))]
    for reference in references:
        include_path = _resolve_path(str(reference), relative_to=exp_path.parent)
        if include_path.with_suffix(".yaml").resolve() == (CONFIG_ROOT / "default.yaml").resolve():
            continue
        included, _ = _load_yaml(include_path)
        merged = OmegaConf.merge(merged, included)

    exp_body = OmegaConf.create(
        {k: v for k, v in exp.items() if k not in {"includes", "defaults"}}
    )
    merged = OmegaConf.merge(merged, exp_body)

    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    OmegaConf.set_readonly(merged, False)
    return merged


def save_config(cfg: DictConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(path))


def to_container(cfg: DictConfig) -> dict:
    """Plain dict, resolved — safe for JSON/W&B logging."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
