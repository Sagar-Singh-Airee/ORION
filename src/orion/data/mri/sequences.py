"""Small, inspectable MRI sequence classifier for series routing."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceType:
    orientation: str
    weighting: str
    fat_saturated: bool


def classify_sequence(description: str, repetition_time: float | None = None, echo_time: float | None = None, inversion_time: float | None = None) -> SequenceType:
    text = description.lower()
    orientation = "sagittal" if "sag" in text else "coronal" if "cor" in text else "axial" if "ax" in text else "unknown"
    if "stir" in text or inversion_time is not None:
        weighting = "stir"
    elif "t1" in text or (repetition_time and repetition_time < 800 and echo_time and echo_time < 30):
        weighting = "t1"
    elif "t2" in text or (echo_time and echo_time > 70):
        weighting = "t2"
    elif "pd" in text or "proton" in text:
        weighting = "pd"
    else:
        weighting = "unknown"
    fat_saturated = any(token in text for token in ("fs", "fat sat", "stir", "dixon", "tirm"))
    return SequenceType(orientation, weighting, fat_saturated)
