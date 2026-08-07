"""Complete model architectures."""
from .image_only import ImageOnlyMILModel
from .multimodal import ORIONMultimodalModel

__all__ = ["ImageOnlyMILModel", "ORIONMultimodalModel"]
