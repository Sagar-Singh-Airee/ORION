"""
DICOM Reader Utility

WHY it exists:
A single MRI sequence consists of multiple 2D DICOM files (slices).
Reading them requires:
1. Identifying all files for a given sequence/series.
2. Sorting them in physical space (not just alphabetically by filename).
3. Extracting the raw pixel arrays and stacking them into a 3D volume.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pydicom
from loguru import logger

def load_dicom_file(filepath: str | Path) -> Optional[pydicom.FileDataset]:
    """
    Safely loads a single DICOM file.
    Returns None if the file is invalid or not a DICOM.
    """
    try:
        # stop_before_pixels=False because we need the image data
        return pydicom.dcmread(str(filepath), stop_before_pixels=False)
    except pydicom.errors.InvalidDicomError:
        logger.warning(f"Invalid DICOM file: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None


def get_series_files(study_dir: str | Path, series_uid: str) -> List[Path]:
    """
    Given a study directory, find all DICOM files belonging to a specific SeriesInstanceUID.
    (In the RSNA dataset, directories might already be organized by series).
    """
    study_dir = Path(study_dir)
    series_files = []
    
    # Iterate through all files in the directory
    for root, _, files in os.walk(study_dir):
        for file in files:
            filepath = Path(root) / file
            # Quick check without loading full pixels
            try:
                ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
                if ds.SeriesInstanceUID == series_uid:
                    series_files.append(filepath)
            except Exception:
                continue
                
    return series_files


def sort_slices(dicom_slices: List[pydicom.FileDataset]) -> List[pydicom.FileDataset]:
    """
    Sorts DICOM slices based on their physical location in 3D space.
    
    WHY: Filenames are often arbitrary (e.g., 1.dcm, 2.dcm might not be adjacent slices).
    We must use the ImagePositionPatient and ImageOrientationPatient tags to
    compute the slice location along the normal vector of the image plane.
    """
    if not dicom_slices:
        return []

    # Ensure all slices have the required tags
    valid_slices = []
    for dcm in dicom_slices:
        if hasattr(dcm, "ImagePositionPatient") and hasattr(dcm, "ImageOrientationPatient"):
            valid_slices.append(dcm)
        else:
            # Fallback: Try InstanceNumber if spatial metadata is missing
            if hasattr(dcm, "InstanceNumber"):
                valid_slices.append(dcm)

    if not valid_slices:
        logger.error("No valid spatial tags or InstanceNumbers found in slices.")
        return dicom_slices

    # If we have spatial tags, calculate projection along the slice normal
    if hasattr(valid_slices[0], "ImagePositionPatient") and hasattr(valid_slices[0], "ImageOrientationPatient"):
        # The orientation is 6 values: (x,y,z) for row vector, (x,y,z) for col vector
        # The cross product gives the normal vector to the plane
        def get_slice_location(dcm: pydicom.FileDataset) -> float:
            pos = np.array(dcm.ImagePositionPatient, dtype=float)
            ori = np.array(dcm.ImageOrientationPatient, dtype=float)
            row_vec = ori[0:3]
            col_vec = ori[3:6]
            normal_vec = np.cross(row_vec, col_vec)
            # Project position onto normal vector
            return np.dot(pos, normal_vec) # type: ignore

        sorted_slices = sorted(valid_slices, key=get_slice_location)
    else:
        # Fallback to InstanceNumber
        sorted_slices = sorted(valid_slices, key=lambda x: int(x.InstanceNumber)) # type: ignore

    return sorted_slices


def load_series_volume(file_paths: List[Path]) -> Tuple[np.ndarray, List[pydicom.FileDataset]]:
    """
    Loads a list of DICOM paths, sorts them, and extracts the 3D volume.
    
    Returns:
        volume: NumPy array of shape (Slices, Height, Width)
        slices: List of the loaded, sorted pydicom datasets
    """
    dicom_slices = []
    for fp in file_paths:
        ds = load_dicom_file(fp)
        if ds is not None:
            dicom_slices.append(ds)

    sorted_slices = sort_slices(dicom_slices)
    
    # Extract pixel arrays
    # Note: Using ds.pixel_array which relies on pydicom pixel handlers
    arrays = []
    for ds in sorted_slices:
        if hasattr(ds, "pixel_array"):
            arrays.append(ds.pixel_array)
            
    if not arrays:
        raise ValueError("No pixel data found in the provided files.")
        
    volume = np.stack(arrays, axis=0)
    return volume, sorted_slices
