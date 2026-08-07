"""
Transform Factory

WHY it exists:
Instead of hardcoding complex augmentation pipelines, this factory reads the
`augmentation.yaml` configuration and dynamically builds the Albumentations pipeline.
This allows us to experiment with different augmentations without changing code.
"""

from typing import Dict, Any, Callable
from loguru import logger
from omegaconf import DictConfig

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def create_train_transforms(config: DictConfig) -> Callable:
    """
    Creates the training data augmentation pipeline based on config.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        logger.error("Albumentations not installed. Using fallback Identity transform.")
        return lambda image=None, **kwargs: {"image": image}
        
    aug_cfg = config.data.augmentation
    if not aug_cfg.get("train", True):
        return create_val_transforms(config)
        
    img_size = config.data.preprocessing.image_size[0]
    transforms = []
    
    # --- Spatial Augmentations ---
    spatial = aug_cfg.get("spatial", {})
    if spatial.get("hflip", {}).get("enabled", False):
        transforms.append(A.HorizontalFlip(p=spatial.hflip.p))
        
    if spatial.get("vflip", {}).get("enabled", False):
        transforms.append(A.VerticalFlip(p=spatial.vflip.p))
        
    if spatial.get("rotation", {}).get("enabled", False):
        transforms.append(A.Rotate(limit=spatial.rotation.limit, p=spatial.rotation.p))
        
    if spatial.get("random_crop", {}).get("enabled", False):
        transforms.append(
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=list(spatial.random_crop.scale),
                ratio=list(spatial.random_crop.ratio),
                p=spatial.random_crop.p
            )
        )
    else:
        transforms.append(A.Resize(height=img_size, width=img_size))

    # --- Intensity Augmentations ---
    intensity = aug_cfg.get("intensity", {})
    if intensity.get("brightness_contrast", {}).get("enabled", False):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=intensity.brightness_contrast.brightness_limit,
                contrast_limit=intensity.brightness_contrast.contrast_limit,
                p=intensity.brightness_contrast.p
            )
        )
        
    if intensity.get("gaussian_noise", {}).get("enabled", False):
        transforms.append(A.GaussNoise(var_limit=list(intensity.gaussian_noise.var_limit), p=intensity.gaussian_noise.p))
        
    if intensity.get("coarse_dropout", {}).get("enabled", False):
        transforms.append(
            A.CoarseDropout(
                max_holes=intensity.coarse_dropout.max_holes,
                max_height=intensity.coarse_dropout.max_height,
                max_width=intensity.coarse_dropout.max_width,
                p=intensity.coarse_dropout.p
            )
        )

    # Convert to PyTorch Tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms, additional_targets={})


def create_val_transforms(config: DictConfig) -> Callable:
    """
    Creates the validation data pipeline (resize and ToTensor only).
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return lambda image=None, **kwargs: {"image": image}
        
    img_size = config.data.preprocessing.image_size[0]
    
    transforms = [
        A.Resize(height=img_size, width=img_size),
        ToTensorV2()
    ]
    
    return A.Compose(transforms)
