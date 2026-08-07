"""
DICOM Metadata Extractor

WHY it exists:
Often we need to route data differently depending on the scanner manufacturer,
magnetic field strength (1.5T vs 3.0T), or sequence type (T1, T2, PD).
This module extracts that metadata for analysis or conditional processing.
"""

import pydicom
from typing import Dict, Any, List

def extract_study_metadata(slices: List[pydicom.FileDataset]) -> Dict[str, Any]:
    """
    Extracts relevant study and series metadata from the first slice.
    Assuming all slices in a series share these properties.
    """
    if not slices:
        return {}
        
    ds = slices[0]
    
    metadata = {
        # Identifiers
        "PatientID": getattr(ds, "PatientID", "UNKNOWN"),
        "StudyInstanceUID": getattr(ds, "StudyInstanceUID", "UNKNOWN"),
        "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", "UNKNOWN"),
        
        # Scanner Info
        "Manufacturer": getattr(ds, "Manufacturer", "UNKNOWN"),
        "ManufacturerModelName": getattr(ds, "ManufacturerModelName", "UNKNOWN"),
        "MagneticFieldStrength": getattr(ds, "MagneticFieldStrength", -1.0),
        
        # Sequence Info
        "SeriesDescription": getattr(ds, "SeriesDescription", "UNKNOWN"),
        "ProtocolName": getattr(ds, "ProtocolName", "UNKNOWN"),
        "ScanningSequence": getattr(ds, "ScanningSequence", "UNKNOWN"),
        "SequenceVariant": getattr(ds, "SequenceVariant", "UNKNOWN"),
        
        # Spatial Info
        "SliceThickness": getattr(ds, "SliceThickness", -1.0),
        "PixelSpacing": getattr(ds, "PixelSpacing", [-1.0, -1.0]),
        "ImageOrientationPatient": getattr(ds, "ImageOrientationPatient", []),
        
        # Volume info
        "NumSlices": len(slices)
    }
    
    return metadata


def identify_sequence_type(series_description: str) -> str:
    """
    Heuristically identifies the MRI sequence type and orientation from the description.
    
    WHY: We might want to train models specifically on Sagittal T2, or
    concatenate Sagittal + Coronal + Axial. The series descriptions are messy
    (e.g., "SAG T2 FS", "t2_tse_sag", "Sagittal PD").
    """
    desc = series_description.lower()
    
    # 1. Identify Orientation
    orientation = "unknown"
    if "sag" in desc:
        orientation = "sagittal"
    elif "cor" in desc:
        orientation = "coronal"
    elif "ax" in desc:
        orientation = "axial"
        
    # 2. Identify Contrast Weighting
    weighting = "unknown"
    if "t1" in desc:
        weighting = "t1"
    elif "t2" in desc:
        weighting = "t2"
    elif "pd" in desc or "proton" in desc:
        weighting = "pd"
    elif "stir" in desc:
        weighting = "stir"
        
    # 3. Identify Fat Saturation
    fat_sat = "no"
    if "fs" in desc or "fat" in desc or "stir" in desc or "dixon" in desc or "tirm" in desc:
        fat_sat = "yes"
        
    return f"{orientation}_{weighting}_fs={fat_sat}"
