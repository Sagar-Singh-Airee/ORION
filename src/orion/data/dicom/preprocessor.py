"""
DICOM Preprocessor

WHY it exists:
Transforms raw DICOM pixel arrays into clean, standardized NumPy arrays ready for
neural networks. Handles VOI-LUT, photometric inversion, and spatial normalization.
"""

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from loguru import logger
from typing import List, Tuple

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
    except Exception as e:
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


def preprocess_volume(volume: np.ndarray, slices: List[pydicom.FileDataset], config: dict) -> np.ndarray:
    """
    End-to-end preprocessing pipeline for a 3D volume.
    """
    # 1. We must process slice-by-slice for DICOM-specific corrections (photometric, VOI)
    processed_slices = []
    
    for i, ds in enumerate(slices):
        img = volume[i].copy()
        
        # Fix inversion
        if config.get("fix_photometric", True):
            img = fix_photometric_interpretation(img, ds)
            
        # Apply clinical windowing
        if config.get("apply_voi_lut", True):
            img = apply_windowing(img, ds)
            
        processed_slices.append(img)
        
    processed_vol = np.stack(processed_slices, axis=0)
    
    # 2. Global volume normalization
    if config.get("normalize", True):
        processed_vol = normalize_intensity(processed_vol)
        
    return processed_vol
