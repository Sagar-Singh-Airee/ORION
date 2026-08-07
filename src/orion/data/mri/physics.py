"""
MRI Physics and Theory

WHY it exists:
To build effective medical AI, one must understand how the image was generated.
MRI does not measure density (like CT). It measures the relaxation times (T1, T2)
of hydrogen protons in a magnetic field after being perturbed by a radiofrequency pulse.
"""

import numpy as np

def bloch_equation_signal(PD: float, T1: float, T2: float, TR: float, TE: float) -> float:
    """
    Computes the theoretical MRI signal intensity based on the simplified Bloch equations
    for a Spin Echo (SE) sequence.

    Args:
        PD: Proton Density (concentration of hydrogen atoms)
        T1: Longitudinal relaxation time (ms)
        T2: Transverse relaxation time (ms)
        TR: Repetition Time (ms) - time between consecutive RF pulses
        TE: Echo Time (ms) - time between RF pulse and signal echo

    Returns:
        Theoretical signal intensity.
        
    WHY:
    Understanding this equation explains why different tissues appear bright or dark.
    - T1 weighting (short TR, short TE): Fat is bright, fluid is dark.
    - T2 weighting (long TR, long TE): Fluid is bright (effusion, edema), fat is dark.
    - PD weighting (long TR, short TE): Good anatomical detail for menisci/ligaments.
    """
    # The signal is proportional to proton density
    # The (1 - exp(-TR/T1)) term governs T1 recovery
    # The exp(-TE/T2) term governs T2 decay
    
    # Avoid division by zero
    if T1 <= 0 or T2 <= 0:
        return 0.0
        
    signal = PD * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)
    return signal

# Typical T1/T2 values (ms) at 1.5 Tesla
# This helps understand what the network is actually "seeing"
TISSUE_PROPERTIES = {
    "fat": {"T1": 250, "T2": 80, "PD": 1.0},
    "muscle": {"T1": 900, "T2": 50, "PD": 0.8},
    "cartilage": {"T1": 1050, "T2": 40, "PD": 0.7},
    "synovial_fluid": {"T1": 2500, "T2": 250, "PD": 1.0},
    "bone_marrow": {"T1": 350, "T2": 60, "PD": 0.9} # mostly fat
}
