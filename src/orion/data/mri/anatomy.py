"""
Knee Anatomy and ROI definitions

WHY it exists:
Instead of forcing the network to search the entire 3D volume, we can use prior
anatomical knowledge to guide it. E.g., the ACL is always central; the medial
meniscus is always on the medial side.
"""
from __future__ import annotations

__all__ = ["KNEE_COMPARTMENTS", "RSNA_LABELS", "compartment_for"]

# Tuples, not lists: these are shared module-level singletons imported all over the
# codebase. A list here is a classic mutable-shared-state hazard — any code that
# accidentally calls .append()/.sort() on it (instead of copying first) would corrupt
# it for every other importer for the lifetime of the process.
KNEE_COMPARTMENTS: dict[str, tuple[str, ...]] = {
    "medial": ("medial_meniscus", "medial_femoral_condyle", "medial_tibial_plateau", "mcl"),
    "lateral": ("lateral_meniscus", "lateral_femoral_condyle", "lateral_tibial_plateau", "lcl"),
    "patellofemoral": ("patella", "trochlear_groove", "patellar_tendon", "quadriceps_tendon"),
    "central": ("acl", "pcl"),
}

# WARNING: this is a SEPARATE list from `orion.data.text.label_extractor.FINDINGS`,
# which is the list actually used to order weak-label extraction, the model's output
# head, and every training/eval/submission script in this codebase (calibrate.py,
# ensemble.py, evaluate.py, train.py, predict.py, knee_mri.py, sampler.py all key off
# FINDINGS). If this list and FINDINGS ever diverge in content or ORDER, anything that
# reads RSNA_LABELS gets silently misaligned against everything that reads FINDINGS —
# same index, different label. Verify these two match (content AND order) before using
# RSNA_LABELS for anything that touches model outputs; consider deriving one from the
# other, or both from one shared constants module, once it's confirmed safe to import
# between them without a cycle.
RSNA_LABELS: tuple[str, ...] = (
    "acl",
    "mcl",
    "medial_meniscus",
    "lateral_meniscus",
    "medial_oa",
    "lateral_oa",
    "pf_oa",
    "effusion",
    "synovitis",
    "bakers_cyst",
    "contusion",
    "fracture",
)

assert len(RSNA_LABELS) == len(set(RSNA_LABELS)), "RSNA_LABELS contains duplicate entries"

# Built once at import time for O(1) lookup rather than scanning every compartment on
# each call — this can plausibly be called per-sample or per-batch by ROI/attention
# guidance logic during training.
_STRUCTURE_TO_COMPARTMENT: dict[str, str] = {
    structure: compartment for compartment, structures in KNEE_COMPARTMENTS.items() for structure in structures
}


def compartment_for(structure: str) -> str | None:
    """Which compartment a named anatomical structure belongs to, or None if unknown."""
    return _STRUCTURE_TO_COMPARTMENT.get(structure)