"""
Weak Label Extractor

WHY it exists:
We have 5,000+ reports but no explicit labels for most of them.
We must programmatically extract the 12 target labels using keyword matching,
negation detection (e.g., "no evidence of meniscal tear"), and Snorkel.
"""

import re
from typing import Dict, List
from loguru import logger

# Simple keyword rules for demonstration
# In production, this uses libraries like `negspacy` (spaCy) or RadLex ontologies
RULES = {
    "acl": {
        "pos": ["acl tear", "anterior cruciate ligament tear", "sprain", "rupture"],
        "neg": ["intact acl", "normal anterior cruciate ligament"]
    },
    "medial_meniscus": {
        "pos": ["medial meniscus tear", "macerated medial meniscus", "complex tear"],
        "neg": ["intact medial meniscus", "unremarkable medial meniscus"]
    },
    "effusion": {
        "pos": ["joint effusion", "fluid", "hemarthrosis"],
        "neg": ["no effusion", "trace fluid"] # Trace is usually considered negative clinically
    }
}

def extract_labels_rule_based(report_text: str) -> Dict[str, float]:
    """
    Extracts probabilistic labels [0.0, 1.0] from text.
    0.0 = confidently negative
    1.0 = confidently positive
    0.5 = unknown / indeterminate
    """
    labels = {k: 0.5 for k in RULES.keys()} # Default to indeterminate
    
    text = report_text.lower()
    
    # Basic NegEx (Negation Extraction) logic
    negation_modifiers = ["no", "without", "denies", "free of", "unremarkable"]
    
    for label, rules in RULES.items():
        # Check positive rules
        for pos_term in rules["pos"]:
            if pos_term in text:
                # Naive check: is there a negation word right before it?
                # e.g., "no acl tear"
                idx = text.find(pos_term)
                window = text[max(0, idx-15):idx]
                
                is_negated = any(n in window for n in negation_modifiers)
                if is_negated:
                    labels[label] = 0.0 # Definitely negative
                else:
                    labels[label] = 1.0 # Definitely positive
                break
                
        # If still indeterminate, check explicit negative rules
        if labels[label] == 0.5:
            for neg_term in rules["neg"]:
                if neg_term in text:
                    labels[label] = 0.0
                    break
                    
    return labels
