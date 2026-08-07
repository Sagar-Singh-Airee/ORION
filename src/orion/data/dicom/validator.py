"""
DICOM Validator

WHY it exists:
Medical datasets are notoriously messy. Series might be missing slices,
contain corrupted files, or have inconsistent voxel spacings.
Validating before training prevents mid-epoch crashes.
"""

import numpy as np
import pydicom
from typing import List, Tuple
from loguru import logger

def validate_series_consistency(slices: List[pydicom.FileDataset]) -> Tuple[bool, List[str]]:
    """
    Checks if a list of DICOM slices forms a consistent 3D volume.
    
    Returns:
        is_valid: bool
        errors: list of error strings
    """
    if not slices:
        return False, ["Empty slice list"]
        
    errors = []
    
    # Check 1: Do they all have the same pixel dimensions?
    base_rows = getattr(slices[0], "Rows", None)
    base_cols = getattr(slices[0], "Columns", None)
    
    for i, ds in enumerate(slices):
        rows = getattr(ds, "Rows", None)
        cols = getattr(ds, "Columns", None)
        if rows != base_rows or cols != base_cols:
            errors.append(f"Inconsistent dimensions at slice {i}: ({rows}, {cols}) vs base ({base_rows}, {base_cols})")
            
    # Check 2: Check for missing slices (gaps in physical space)
    # If spatial info exists, compute the distance between consecutive slices
    if hasattr(slices[0], "ImagePositionPatient") and hasattr(slices[0], "ImageOrientationPatient"):
        def get_slice_location(dcm: pydicom.FileDataset) -> float:
            pos = np.array(dcm.ImagePositionPatient, dtype=float)
            ori = np.array(dcm.ImageOrientationPatient, dtype=float)
            normal_vec = np.cross(ori[0:3], ori[3:6])
            return np.dot(pos, normal_vec) # type: ignore
            
        locs = [get_slice_location(ds) for ds in slices]
        # Calculate diffs between consecutive slices
        diffs = np.diff(np.sort(locs))
        
        if len(diffs) > 0:
            median_spacing = np.median(np.abs(diffs))
            # If any gap is > 1.5x the median spacing, it suggests a missing slice
            if median_spacing > 1e-6 and np.any(np.abs(diffs) > 1.5 * median_spacing):
                errors.append("Large gap between slices detected; possible missing slice.")
                
    return len(errors) == 0, errors
