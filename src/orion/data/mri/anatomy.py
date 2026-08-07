"""
Knee Anatomy and ROI definitions

WHY it exists:
Instead of forcing the network to search the entire 3D volume, we can use prior
anatomical knowledge to guide it. E.g., the ACL is always central; the medial
meniscus is always on the medial side.
"""

KNEE_COMPARTMENTS = {
    "medial": ["medial_meniscus", "medial_femoral_condyle", "medial_tibial_plateau", "mcl"],
    "lateral": ["lateral_meniscus", "lateral_femoral_condyle", "lateral_tibial_plateau", "lcl"],
    "patellofemoral": ["patella", "trochlear_groove", "patellar_tendon", "quadriceps_tendon"],
    "central": ["acl", "pcl"]
}

# 12 Target Labels Mapping for RSNA 2026
RSNA_LABELS = [
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
    "fracture"
]
