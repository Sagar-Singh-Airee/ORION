"""
MRI Physics and Theory
======================

WHY IT EXISTS
-------------
To build effective medical AI, one must understand how the image was generated.

MRI does not directly measure tissue density like CT. MRI signal depends on
properties such as proton density (PD), longitudinal relaxation (T1), and
transverse relaxation (T2), together with the pulse-sequence parameters.

This module implements a simplified Spin Echo (SE) MRI signal model:

    S = PD * (1 - exp(-TR / T1)) * exp(-TE / T2)

This is an educational/theoretical model, not a full Bloch-equation simulator.
Actual MRI signal depends on many additional factors, including field strength,
sequence design, flip angle, coil sensitivity, tissue composition, and scanner
characteristics.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

import numpy as np

__all__ = [
    "TissueProperties",
    "TISSUE_PROPERTIES",
    "spin_echo_signal",
    "signal_for_tissue",
]


class TissueProperties(NamedTuple):
    """
    Basic MRI tissue properties used by the simplified signal model.

    Attributes
    ----------
    pd:
        Relative proton density.
    t1:
        Longitudinal relaxation time in milliseconds.
    t2:
        Transverse relaxation time in milliseconds.
    """

    pd: float
    t1: float
    t2: float


# ---------------------------------------------------------------------------
# Simplified Spin Echo Signal Model
# ---------------------------------------------------------------------------

def spin_echo_signal(
    PD: float,
    T1: float,
    T2: float,
    TR: float,
    TE: float,
) -> float:
    """
    Compute theoretical MRI signal intensity for a simplified Spin Echo
    sequence.

    Parameters
    ----------
    PD:
        Relative proton density.

    T1:
        Longitudinal relaxation time in milliseconds.

    T2:
        Transverse relaxation time in milliseconds.

    TR:
        Repetition time in milliseconds.

    TE:
        Echo time in milliseconds.

    Returns
    -------
    float
        Theoretical relative MRI signal intensity.

    Notes
    -----
    The simplified signal equation is:

        S = PD * (1 - exp(-TR / T1)) * exp(-TE / T2)

    The individual terms represent:

    - PD:
        Proton density contribution.

    - (1 - exp(-TR / T1)):
        Longitudinal T1 recovery.

    - exp(-TE / T2):
        Transverse T2 decay.

    Approximate weighting behavior:

    T1-weighted:
        Short TR + short TE
        Fat tends to appear bright and fluid relatively dark.

    T2-weighted:
        Long TR + long TE
        Fluid tends to appear bright.

    Proton-density-weighted:
        Long TR + short TE
        Useful for anatomical detail, including structures such as
        menisci and ligaments.

    Important
    ---------
    This is a simplified theoretical model. It does not represent every
    physical process involved in real MRI acquisition.
    """

    # Validate proton density.
    if not np.isfinite(PD):
        raise ValueError(f"PD must be finite, got {PD}")

    if PD < 0:
        raise ValueError(
            f"PD (proton density) must be non-negative, got {PD}"
        )

    # Relaxation times must be physically meaningful.
    if not np.isfinite(T1) or not np.isfinite(T2):
        raise ValueError(
            f"T1 and T2 must be finite, got T1={T1}, T2={T2}"
        )

    if T1 <= 0 or T2 <= 0:
        raise ValueError(
            "T1 and T2 must be positive relaxation times, "
            f"got T1={T1}, T2={T2}"
        )

    # TR and TE cannot be negative.
    if not np.isfinite(TR) or not np.isfinite(TE):
        raise ValueError(
            f"TR and TE must be finite, got TR={TR}, TE={TE}"
        )

    if TR < 0 or TE < 0:
        raise ValueError(
            f"TR and TE must be non-negative, got TR={TR}, TE={TE}"
        )

    # -----------------------------------------------------------------------
    # Simplified Spin Echo signal equation
    #
    # PD                 -> proton density
    # (1 - exp(-TR/T1))  -> longitudinal T1 recovery
    # exp(-TE/T2)        -> transverse T2 decay
    # -----------------------------------------------------------------------

    signal = (
        PD
        * (1.0 - np.exp(-TR / T1))
        * np.exp(-TE / T2)
    )

    # Explicit conversion guarantees a normal Python float is returned.
    return float(signal)


# ---------------------------------------------------------------------------
# Typical Tissue Properties
# ---------------------------------------------------------------------------
#
# These are illustrative approximate values around 1.5 Tesla.
#
# Actual T1/T2 values vary with:
#   - Magnetic field strength
#   - Scanner manufacturer
#   - Pulse sequence
#   - Acquisition parameters
#   - Tissue composition
#   - Patient characteristics
#   - Measurement methodology
#
# Therefore, these values should NOT be treated as universal ground truth.
# ---------------------------------------------------------------------------

TISSUE_PROPERTIES: Mapping[str, TissueProperties] = MappingProxyType(
    {
        "fat": TissueProperties(
            pd=1.0,
            t1=250.0,
            t2=80.0,
        ),
        "muscle": TissueProperties(
            pd=0.8,
            t1=900.0,
            t2=50.0,
        ),
        "cartilage": TissueProperties(
            pd=0.7,
            t1=1050.0,
            t2=40.0,
        ),
        "synovial_fluid": TissueProperties(
            pd=1.0,
            t1=2500.0,
            t2=250.0,
        ),
        "bone_marrow": TissueProperties(
            pd=0.9,
            t1=350.0,
            t2=60.0,
        ),
    }
)


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def signal_for_tissue(
    tissue: str,
    TR: float,
    TE: float,
) -> float:
    """
    Compute the theoretical MRI signal for a named tissue.

    Parameters
    ----------
    tissue:
        Tissue name from TISSUE_PROPERTIES.

    TR:
        Repetition time in milliseconds.

    TE:
        Echo time in milliseconds.

    Returns
    -------
    float
        Theoretical relative MRI signal intensity.

    Examples
    --------
    Compare fat and synovial fluid using a T1-weighted-like acquisition:

        signal_for_tissue("fat", TR=500, TE=15)
        signal_for_tissue("synovial_fluid", TR=500, TE=15)
    """

    if tissue not in TISSUE_PROPERTIES:
        available = ", ".join(sorted(TISSUE_PROPERTIES))

        raise ValueError(
            f"Unknown tissue {tissue!r}. "
            f"Expected one of: {available}"
        )

    properties = TISSUE_PROPERTIES[tissue]

    return spin_echo_signal(
        properties.pd,
        properties.t1,
        properties.t2,
        TR=TR,
        TE=TE,
    )