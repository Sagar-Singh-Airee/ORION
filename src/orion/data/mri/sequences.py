"""
MRI Sequence Classification and Series Routing
================================================

WHY IT EXISTS
-------------
MRI datasets commonly contain many series for the same study:

    Sagittal PD FS
    Sagittal T2
    Coronal PD FS
    Axial T2 FS
    Localizers
    Dixon
    STIR
    T1
    etc.

Before preprocessing or model inference, these series need to be identified
and routed consistently.

This module provides a small, inspectable heuristic classifier that converts
free-text series descriptions and optional DICOM timing metadata into a
structured SequenceType object.

IMPORTANT
---------
This is a routing/classification heuristic, NOT a definitive MRI sequence
parser.

MRI protocols vary across scanners, manufacturers, institutions, field
strengths, and acquisition strategies. Therefore:

    description-based evidence
        >
    explicit sequence terminology
        >
    acquisition timing heuristics

should be preferred when determining sequence characteristics.

The classifier is intentionally conservative: when evidence is insufficient,
it returns "unknown" rather than inventing a classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


__all__ = [
    "SequenceType",
    "classify_sequence",
]


# ---------------------------------------------------------------------------
# Structured sequence representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceType:
    """
    Structured representation of an MRI sequence classification.

    Attributes
    ----------
    orientation:
        Anatomical orientation:
        "sagittal", "coronal", "axial", or "unknown".

    weighting:
        Approximate contrast weighting:
        "t1", "t2", "pd", "stir", or "unknown".

    fat_saturated:
        Whether the sequence appears to use fat suppression.

    sequence_family:
        Broad sequence family such as:
        "spin_echo", "fast_spin_echo", "inversion_recovery",
        "gradient_echo", "dixon", "unknown".

    confidence:
        Heuristic confidence from 0.0 to 1.0.

    evidence:
        Human-readable explanation of the strongest classification signals.
    """

    orientation: str
    weighting: str
    fat_saturated: bool
    sequence_family: str
    confidence: float
    evidence: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize_description(description: str) -> str:
    """
    Normalize a DICOM series description for robust matching.
    """

    if not isinstance(description, str):
        raise TypeError(
            "description must be a string, "
            f"got {type(description).__name__}"
        )

    return " ".join(description.lower().strip().split())


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    """Return True if any token occurs in the normalized description."""

    return any(token in text for token in tokens)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_timing(
    name: str,
    value: float | None,
) -> None:
    """Validate optional DICOM timing parameters."""

    if value is None:
        return

    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a number or None, "
            f"got {type(value).__name__}"
        )

    if not isfinite(float(value)):
        raise ValueError(
            f"{name} must be finite, got {value}"
        )

    if value < 0:
        raise ValueError(
            f"{name} must be non-negative, got {value}"
        )


# ---------------------------------------------------------------------------
# Orientation classification
# ---------------------------------------------------------------------------


def _classify_orientation(text: str) -> tuple[str, float, tuple[str, ...]]:
    """
    Determine anatomical orientation from sequence description.

    Returns
    -------
    orientation, confidence, evidence
    """

    sagittal_tokens = (
        "sagittal",
        "sag",
    )

    coronal_tokens = (
        "coronal",
        "cor",
    )

    axial_tokens = (
        "axial",
        "ax",
        "transverse",
        "tra",
    )

    if _contains_any(text, sagittal_tokens):
        return (
            "sagittal",
            0.98,
            ("explicit sagittal terminology",),
        )

    if _contains_any(text, coronal_tokens):
        return (
            "coronal",
            0.98,
            ("explicit coronal terminology",),
        )

    if _contains_any(text, axial_tokens):
        return (
            "axial",
            0.98,
            ("explicit axial/transverse terminology",),
        )

    return (
        "unknown",
        0.0,
        ("no reliable orientation terminology",),
    )


# ---------------------------------------------------------------------------
# Fat-suppression classification
# ---------------------------------------------------------------------------


def _classify_fat_suppression(
    text: str,
) -> tuple[bool, tuple[str, ...]]:
    """
    Detect common fat-suppression terminology.
    """

    indicators = (
        "fat sat",
        "fat-sat",
        "fatsat",
        "fat saturation",
        "stir",
        "tirm",
        "dixon",
    )

    matches = tuple(
        token for token in indicators
        if token in text
    )

    if matches:
        return True, (
            f"fat-suppression indicator(s): {', '.join(matches)}",
        )

    return False, ()


# ---------------------------------------------------------------------------
# Sequence-family classification
# ---------------------------------------------------------------------------


def _classify_sequence_family(
    text: str,
) -> tuple[str, float, tuple[str, ...]]:
    """
    Identify the broad MRI sequence family.
    """

    if _contains_any(
        text,
        (
            "stir",
            "tirm",
            "inversion recovery",
        ),
    ):
        return (
            "inversion_recovery",
            0.98,
            ("explicit inversion-recovery terminology",),
        )

    if _contains_any(
        text,
        (
            "dixon",
            "water fat",
            "water/fat",
        ),
    ):
        return (
            "dixon",
            0.98,
            ("explicit Dixon terminology",),
        )

    if _contains_any(
        text,
        (
            "gre",
            "gradient echo",
            "gradient-echo",
            "spoiled gradient",
            "spgr",
            "flash",
        ),
    ):
        return (
            "gradient_echo",
            0.95,
            ("explicit gradient-echo terminology",),
        )

    if _contains_any(
        text,
        (
            "fse",
            "fast spin echo",
            "fast-spin-echo",
            "tse",
            "turbo spin echo",
            "turbo-spin-echo",
        ),
    ):
        return (
            "fast_spin_echo",
            0.95,
            ("explicit fast/turbo spin-echo terminology",),
        )

    if _contains_any(
        text,
        (
            "spin echo",
            "spin-echo",
            "se ",
        ),
    ):
        return (
            "spin_echo",
            0.90,
            ("explicit spin-echo terminology",),
        )

    return (
        "unknown",
        0.0,
        (),
    )


# ---------------------------------------------------------------------------
# Weighting classification
# ---------------------------------------------------------------------------


def _classify_weighting(
    text: str,
    repetition_time: float | None,
    echo_time: float | None,
) -> tuple[str, float, tuple[str, ...]]:
    """
    Determine approximate image weighting.

    Explicit sequence terminology is always preferred over timing heuristics.
    """

    # ---------------------------------------------------------------
    # Explicit STIR / inversion recovery
    # ---------------------------------------------------------------

    if _contains_any(
        text,
        (
            "stir",
            "tirm",
            "inversion recovery",
        ),
    ):
        return (
            "stir",
            0.99,
            ("explicit STIR/TIRM terminology",),
        )

    # ---------------------------------------------------------------
    # Explicit T1 terminology
    # ---------------------------------------------------------------

    if _contains_any(
        text,
        (
            "t1",
            "t1w",
            "t1 weighted",
            "t1-weighted",
        ),
    ):
        return (
            "t1",
            0.99,
            ("explicit T1 terminology",),
        )

    # ---------------------------------------------------------------
    # Explicit T2 terminology
    # ---------------------------------------------------------------

    if _contains_any(
        text,
        (
            "t2",
            "t2w",
            "t2 weighted",
            "t2-weighted",
        ),
    ):
        return (
            "t2",
            0.99,
            ("explicit T2 terminology",),
        )

    # ---------------------------------------------------------------
    # Explicit proton-density terminology
    # ---------------------------------------------------------------

    if _contains_any(
        text,
        (
            "pd",
            "pdw",
            "proton density",
            "proton-density",
        ),
    ):
        return (
            "pd",
            0.99,
            ("explicit proton-density terminology",),
        )

    # ---------------------------------------------------------------
    # Conservative timing heuristics
    # ---------------------------------------------------------------

    if repetition_time is not None and echo_time is not None:
        # These are approximate routing heuristics only.
        #
        # They intentionally require both TR and TE because either parameter
        # alone is insufficient to reliably determine weighting.

        if (
            repetition_time > 0
            and repetition_time < 800
            and echo_time > 0
            and echo_time < 30
        ):
            return (
                "t1",
                0.65,
                ("approximate T1 timing pattern from TR/TE",),
            )

        if (
            repetition_time >= 1500
            and echo_time >= 60
        ):
            return (
                "t2",
                0.65,
                ("approximate T2 timing pattern from TR/TE",),
            )

        if (
            repetition_time >= 1500
            and echo_time < 40
        ):
            return (
                "pd",
                0.55,
                ("possible proton-density timing pattern from TR/TE",),
            )

    return (
        "unknown",
        0.0,
        ("insufficient evidence for reliable weighting",),
    )


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------


def classify_sequence(
    description: str,
    repetition_time: float | None = None,
    echo_time: float | None = None,
    inversion_time: float | None = None,
) -> SequenceType:
    """
    Classify an MRI sequence for downstream series routing.

    Parameters
    ----------
    description:
        DICOM SeriesDescription or similar free-text sequence description.

    repetition_time:
        Repetition time (TR), milliseconds.

    echo_time:
        Echo time (TE), milliseconds.

    inversion_time:
        Inversion time (TI), milliseconds.

    Returns
    -------
    SequenceType
        Structured sequence classification.

    Examples
    --------
    >>> classify_sequence("Sagittal PD FS")
    SequenceType(
        orientation="sagittal",
        weighting="pd",
        fat_saturated=True,
        ...
    )

    >>> classify_sequence(
    ...     "Axial T2 FS",
    ...     repetition_time=4000,
    ...     echo_time=80,
    ... )
    """

    _validate_timing("repetition_time", repetition_time)
    _validate_timing("echo_time", echo_time)
    _validate_timing("inversion_time", inversion_time)

    text = _normalize_description(description)

    # ---------------------------------------------------------------
    # Individual classifications
    # ---------------------------------------------------------------

    orientation, orientation_confidence, orientation_evidence = (
        _classify_orientation(text)
    )

    fat_saturated, fat_evidence = _classify_fat_suppression(text)

    sequence_family, family_confidence, family_evidence = (
        _classify_sequence_family(text)
    )

    weighting, weighting_confidence, weighting_evidence = (
        _classify_weighting(
            text,
            repetition_time,
            echo_time,
        )
    )

    # ---------------------------------------------------------------
    # Confidence aggregation
    # ---------------------------------------------------------------
    #
    # Weighting and orientation are the most important characteristics
    # for downstream image routing.
    # ---------------------------------------------------------------

    confidence_components = [
        confidence
        for confidence in (
            orientation_confidence,
            weighting_confidence,
            family_confidence,
        )
        if confidence > 0
    ]

    if confidence_components:
        confidence = sum(confidence_components) / len(
            confidence_components
        )
    else:
        confidence = 0.0

    # Having explicit timing metadata can slightly strengthen a
    # classification, but timing alone never receives high confidence.
    if (
        weighting_confidence >= 0.55
        and repetition_time is not None
        and echo_time is not None
    ):
        confidence = min(1.0, confidence + 0.02)

    evidence = (
        orientation_evidence
        + weighting_evidence
        + family_evidence
        + fat_evidence
    )

    return SequenceType(
        orientation=orientation,
        weighting=weighting,
        fat_saturated=fat_saturated,
        sequence_family=sequence_family,
        confidence=round(confidence, 3),
        evidence=evidence,
    )