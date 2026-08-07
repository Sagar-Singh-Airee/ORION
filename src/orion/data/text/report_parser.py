"""
Radiology Report Parser

WHY it exists:
Radiology reports contain the ground truth (weak labels) for 4,900+ studies.
However, they are free-text. We must parse them into structured sections
(Findings, Impression) before passing them to a text encoder or label extractor.
"""

import re
from typing import Dict, Optional

def extract_sections(report_text: str) -> Dict[str, str]:
    """
    Parses a radiology report into sections like 'clinical_indication',
    'findings', and 'impression'.
    
    WHY: The 'Impression' (or Conclusion) section usually contains the
    most concentrated diagnostic information. The 'Findings' section
    contains detailed compartment-by-compartment descriptions.
    """
    sections = {
        "clinical_indication": "",
        "technique": "",
        "findings": "",
        "impression": ""
    }
    
    if not isinstance(report_text, str):
        return sections
        
    text = report_text.lower()
    
    # Common headers
    indication_patterns = ["indication:", "clinical indication:", "history:"]
    technique_patterns = ["technique:", "procedure:"]
    findings_patterns = ["findings:"]
    impression_patterns = ["impression:", "conclusion:", "opinion:"]
    
    # Very basic regex parsing (in practice, this requires a more robust state machine)
    # This splits the document by common ALL CAPS or Title Case headers.
    
    # Heuristic parsing
    lines = text.split('\n')
    current_section = None
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
            
        # Check if line is a header
        is_header = False
        for p in indication_patterns:
            if line_clean.startswith(p):
                current_section = "clinical_indication"
                sections[current_section] += line_clean[len(p):] + " "
                is_header = True
                break
        
        if is_header: continue
        
        for p in findings_patterns:
            if line_clean.startswith(p):
                current_section = "findings"
                sections[current_section] += line_clean[len(p):] + " "
                is_header = True
                break
                
        if is_header: continue
        
        for p in impression_patterns:
            if line_clean.startswith(p):
                current_section = "impression"
                sections[current_section] += line_clean[len(p):] + " "
                is_header = True
                break
                
        if is_header: continue
        
        # If not a header, append to current section
        if current_section:
            sections[current_section] += line_clean + " "
            
    # Clean up whitespace
    for k in sections:
        sections[k] = sections[k].strip()
        
    return sections
