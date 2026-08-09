"""
DICOM Preprocessor

WHY it exists:
Transforms raw DICOM pixel arrays into clean, standardized NumPy arrays ready for
neural networks. Handles VOI-LUT, photometric inversion, and spatial normalization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
try:
    from scipy.ndimage import zoom
except ImportError:  # pragma: no cover - scipy is a project dependency
    zoom = None


def _get(config: Any, key: str, default: Any = None) -> Any:
    """Read a key from dicts and OmegaConf objects alike."""
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def fix_photometric_interpretation(image: np.ndarray, dicom_dataset: pydicom.FileDataset) -> np.ndarray:
    """
    Fixes photometric inversion.
    
    WHY: 
    DICOM uses `PhotometricInterpretation` to define how pixel values map to brightness.
    - MONOCHROME2: 0 is black, max is white (Standard for most deep learning).
    - MONOCHROME1: 0 is white, max is black (Inverted).
    If we don't invert MONOCHROME1, the network sees negative images.
    """
    pi = dicom_dataset.get("PhotometricInterpretation", "")
    
    if pi == "MONOCHROME1":
        # Invert the image
        max_val = np.max(image)
        image = max_val - image
        
    return image


def apply_windowing(image: np.ndarray, dicom_dataset: pydicom.FileDataset) -> np.ndarray:
    """
    Applies Value of Interest (VOI) Look-Up Table (LUT) or Windowing.
    
    WHY:
    Raw MRI pixels might range from 0 to 4000. But the clinically meaningful
    contrast might only be between 100 and 500. The scanner embeds this preferred
    "window" in the DICOM tags.
    """
    # Try pydicom's built-in utility first (handles both LUT tables and Window Center/Width)
    try:
        windowed = apply_voi_lut(image, dicom_dataset)
        return windowed
    except Exception:
        # Fallback to manual windowing if tags are missing or malformed
        # Usually MRI doesn't strictly need VOI LUT as much as CT (HU), 
        # but it's safe to attempt.
        if hasattr(dicom_dataset, "WindowCenter") and hasattr(dicom_dataset, "WindowWidth"):
            center = dicom_dataset.WindowCenter
            width = dicom_dataset.WindowWidth
            
            # Handle multiple windows (take the first one)
            if isinstance(center, pydicom.multival.MultiValue):
                center = center[0]
            if isinstance(width, pydicom.multival.MultiValue):
                width = width[0]
                
            center = float(center)
            width = float(width)
            
            lower = center - (width / 2.0)
            upper = center + (width / 2.0)
            
            windowed = np.clip(image, lower, upper)
            return windowed
        
        # If no windowing info, just return the raw image
        return image


def normalize_intensity(volume: np.ndarray, method: str = "percentile", 
                        p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """
    Normalizes the intensity of the volume.
    
    WHY:
    MRI signal intensities are relative, not absolute. Normalizing removes
    scanner-specific biases. We use percentiles to avoid extreme outlier pixels
    (like artifacts) destroying the dynamic range.
    """
    volume = volume.astype(np.float32)
    
    if method == "percentile":
        low = np.percentile(volume, p_low)
        high = np.percentile(volume, p_high)
        
        # Avoid division by zero
        if high - low < 1e-6:
            return np.zeros_like(volume)
            
        # Clip to percentiles and scale to [0, 1]
        volume = np.clip(volume, low, high)
        volume = (volume - low) / (high - low)
        
    elif method == "zscore":
        mean = np.mean(volume)
        std = np.std(volume)
        if std < 1e-6:
            return np.zeros_like(volume)
        volume = (volume - mean) / std
        
    elif method == "global_minmax":
        min_val = np.min(volume)
        max_val = np.max(volume)
        if max_val - min_val < 1e-6:
            return np.zeros_like(volume)
        volume = (volume - min_val) / (max_val - min_val)
        
    return volume


def select_slice_indices(num_slices: int, target_slices: int, strategy: str = "uniform") -> np.ndarray:
    """Select deterministic, ordered slice indices without duplicate endpoints."""
    if num_slices <= 0 or target_slices <= 0:
        return np.empty(0, dtype=np.int64)
    if strategy == "all" or num_slices <= target_slices:
        return np.arange(num_slices, dtype=np.int64)
    if strategy == "center":
        start = max(0, (num_slices - target_slices) // 2)
        return np.arange(start, start + target_slices, dtype=np.int64)
    if strategy != "uniform":
        raise ValueError(f"Unknown slice-selection strategy: {strategy}")
    # Rounding can produce duplicate positions for close source/target sizes.
    indices = np.linspace(0, num_slices - 1, target_slices).round().astype(np.int64)
    return np.maximum.accumulate(indices)


def resize_volume(volume: np.ndarray, image_size: tuple[int, int] | list[int]) -> np.ndarray:
    """Resize in-plane dimensions while preserving slice count and float precision."""
    target_h, target_w = (int(image_size[0]), int(image_size[1]))
    if volume.ndim != 3:
        raise ValueError(f"Expected (S, H, W) volume, got {volume.shape}")
    if volume.shape[1:] == (target_h, target_w):
        return volume.astype(np.float32, copy=False)
    if zoom is None:
        raise ImportError("scipy is required for resizing DICOM volumes")
    factors = (1.0, target_h / volume.shape[1], target_w / volume.shape[2])
    return zoom(volume, factors, order=1, prefilter=False).astype(np.float32, copy=False)


def preprocess_volume(
    volume: np.ndarray,
    slices: List[pydicom.FileDataset] | None,
    config: Any,
) -> np.ndarray:
    """
    End-to-end preprocessing pipeline for a 3D volume.
    """
    # 1. We must process slice-by-slice for DICOM-specific corrections (photometric, VOI)
    if volume.ndim != 3 or volume.shape[0] == 0:
        raise ValueError(f"Expected a non-empty (S, H, W) volume, got {volume.shape}")
    if slices is not None and len(slices) != len(volume):
        raise ValueError("Number of DICOM datasets must match volume slice count")

    fix_photometric = _get(config, "fix_photometric_interpretation", _get(config, "fix_photometric", True))
    apply_voi = _get(config, "apply_voi_lut", True)
    if slices is None or not (fix_photometric or apply_voi):
        processed_vol = volume.astype(np.float32, copy=True)
    else:
        processed_slices = []
        for image, dataset in zip(volume, slices, strict=True):
            processed = image.copy()
            if fix_photometric:
                processed = fix_photometric_interpretation(processed, dataset)
            if apply_voi:
                processed = apply_windowing(processed, dataset)
            processed_slices.append(processed)
        processed_vol = np.stack(processed_slices, axis=0)
    
    # 2. Global volume normalization
    norm_cfg = _get(config, "normalization", config)
    method = _get(norm_cfg, "method", "percentile")
    if _get(config, "normalize", method != "none") and method != "none":
        processed_vol = normalize_intensity(
            processed_vol,
            method=method,
            p_low=float(_get(norm_cfg, "percentile_low", 0.5)),
            p_high=float(_get(norm_cfg, "percentile_high", 99.5)),
        )
        target_range = _get(norm_cfg, "target_range", (0.0, 1.0))
        if method == "percentile" and tuple(target_range) != (0.0, 1.0):
            low, high = float(target_range[0]), float(target_range[1])
            processed_vol = processed_vol * (high - low) + low

    output_cfg = _get(config, "output", {})
    image_size = _get(output_cfg, "image_size", _get(config, "image_size", None))
    if image_size is not None:
        processed_vol = resize_volume(processed_vol, image_size)
        
    return processed_vol.astype(np.float32, copy=False)
