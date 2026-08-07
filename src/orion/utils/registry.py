"""
Registry Pattern Utility

WHY it exists:
Deep learning codebases often become a tangled mess of `if/elif` statements:
  if model_name == "resnet50": ...
  elif model_name == "swin": ...
  
A Registry pattern allows us to decouple definition from instantiation.
Components (models, losses, datasets) register themselves using decorators.
The config file simply specifies the string name, and the factory instantiates it.
"""

from typing import Dict, Any, Type, Callable
from loguru import logger

class Registry:
    """
    A simple registry to map string names to classes or functions.
    """
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable:
        """
        Decorator to register a class or function.
        If name is not provided, uses the class/function name.
        """
        def decorator(obj: Any) -> Any:
            registry_name = name if name is not None else obj.__name__
            if registry_name in self._registry:
                logger.warning(f"Overwriting existing registry entry '{registry_name}' in {self.name}")
            self._registry[registry_name] = obj
            return obj
        return decorator

    def get(self, name: str) -> Any:
        """Retrieves an object from the registry by name."""
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in registry '{self.name}'. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def build(self, name: str, **kwargs: Any) -> Any:
        """Builds an instance of the registered class with provided kwargs."""
        obj_cls = self.get(name)
        return obj_cls(**kwargs)

    def contains(self, name: str) -> bool:
        return name in self._registry

    def __str__(self) -> str:
        return f"Registry({self.name}): {list(self._registry.keys())}"


# Global registries
MODELS = Registry("models")
BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")
FUSION = Registry("fusion")
LOSSES = Registry("losses")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
