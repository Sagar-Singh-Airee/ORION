"""Weak label extraction from free-text radiology reports.

With ~58 expert-labeled studies vs 4,900+ report-only studies, this module is what
turns the report-only studies into usable (noisy) training signal. Output per finding
is ternary: 1 = positive mention, 0 = explicitly negated, -1 = not mentioned / uncertain
(treat as missing in the loss, not as negative — collapsing -1 into 0 silently biases
every rare finding toward "absent").

Design: per-finding regex triggers + a NegEx-style negation/uncertainty scope window
(negation cue -> N tokens forward, or backward for post-negation cues like "ruled out").
This is intentionally a labeling *function* per finding, not a black-box classifier,
so weak_supervision/label_model.py can later combine it with other labeling sources
(e.g. a second extractor, or the expert labels) via Snorkel-style aggregation without
having to trust any single function.

Assumes input text is English — data/text/translator.py must run first for the
9-language report set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FINDINGS = [
    "acl", "mcl", "medial_meniscus", "lateral_meniscus",
    "medial_oa", "lateral_oa", "pf_oa", "effusion",
    "synovitis", "bakers_cyst", "contusion", "fracture",
]

# Each pattern is matched case-insensitively against sentence text.
FINDING_PATTERNS: dict[str, list[str]] = {
    "acl": [r"\bacl\b", r"anterior cruciate ligament"],
    "mcl": [r"\bmcl\b", r"medial collateral ligament"],
    "medial_meniscus": [r"medial meniscus", r"medial meniscal"],
    "lateral_meniscus": [r"lateral meniscus", r"lateral meniscal"],
    "medial_oa": [r"medial compartment.{0,20}(osteoarthrit|degenerat|joint space narrowing)",
                  r"medial.{0,15}osteoarthrit"],
    "lateral_oa": [r"lateral compartment.{0,20}(osteoarthrit|degenerat|joint space narrowing)",
                   r"lateral.{0,15}osteoarthrit"],
    "pf_oa": [r"patellofemoral.{0,20}(osteoarthrit|degenerat|joint space narrowing|compartment)"],
    "effusion": [r"joint effusion", r"knee effusion", r"\beffusion\b"],
    "synovitis": [r"synovitis", r"synovial (thickening|proliferation|hypertrophy)"],
    "bakers_cyst": [r"baker'?s? cyst", r"popliteal cyst"],
    "contusion": [r"bone (bruis|contusion)", r"\bcontusion\b", r"marrow edema"],
    "fracture": [r"\bfracture\b", r"\bfx\b"],
}

# NegEx-lite cue lists (subset tuned for radiology phrasing).
PRE_NEGATION = [
    "no evidence of", "no signs of", "without evidence of", "no ", "negative for",
    "denies", "absent", "rules? out", "free of",
]
POST_NEGATION = ["is ruled out", "was excluded", "not seen", "not identified", "not present"]
UNCERTAINTY = [
    "cannot exclude", "cannot rule out", "possible", "probable", "suspected",
    "suggestive of", "may represent", "equivocal", "questionable", "differential includes",
    "not fully excluded",
]

_PRE_NEG_RE = re.compile(r"(?:%s)\s*[:\-]?\s*$" % "|".join(PRE_NEGATION), re.IGNORECASE)
_POST_NEG_RE = re.compile(r"^\s*(?:%s)" % "|".join(POST_NEGATION), re.IGNORECASE)
_UNCERTAIN_RE = re.compile("|".join(UNCERTAINTY), re.IGNORECASE)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;\n])\s+")
_NEGATION_WINDOW_CHARS = 60  # lookback/lookahead window around a match, in characters


@dataclass
class ExtractionResult:
    labels: dict[str, int] = field(default_factory=dict)     # finding -> {-1, 0, 1}
    evidence: dict[str, str] = field(default_factory=dict)   # finding -> matched sentence


def _label_sentence(sentence: str, span_start: int, span_end: int) -> int:
    """Given a match's character span within `sentence`, decide 1/0/-1."""
    before = sentence[max(0, span_start - _NEGATION_WINDOW_CHARS):span_start]
    after = sentence[span_end:span_end + _NEGATION_WINDOW_CHARS]

    if _PRE_NEG_RE.search(before):
        return 0
    if _POST_NEG_RE.match(after):
        return 0
    if _UNCERTAIN_RE.search(before) or _UNCERTAIN_RE.search(after):
        return -1
    return 1


def extract_labels(report_text: str) -> ExtractionResult:
    """Extract ternary weak labels for all 12 findings from one English report."""
    result = ExtractionResult()
    if not report_text or not report_text.strip():
        return result

    sentences = _SENTENCE_SPLIT_RE.split(report_text)

    for finding, patterns in FINDING_PATTERNS.items():
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
        best_label: int | None = None
        best_sentence: str | None = None

        for sentence in sentences:
            for pat in compiled:
                m = pat.search(sentence)
                if not m:
                    continue
                label = _label_sentence(sentence, m.start(), m.end())
                # Positive evidence anywhere wins over negation/uncertainty found elsewhere
                # in the report — a single confirmed mention outweighs a hedge in another
                # sentence (e.g. impression restates findings section positively).
                if best_label is None or label == 1:
                    best_label, best_sentence = label, sentence.strip()
                if label == 1:
                    break
            if best_label == 1:
                break

        result.labels[finding] = best_label if best_label is not None else -1
        if best_sentence is not None:
            result.evidence[finding] = best_sentence

    return result


def extract_labels_batch(report_texts: list[str]) -> list[ExtractionResult]:
    return [extract_labels(t) for t in report_texts]


def labels_to_vector(result: ExtractionResult, missing_as: int = -1) -> list[int]:
    """Fixed-order 12-vector for the FINDINGS list, for dataset/collate consumption."""
    return [result.labels.get(f, missing_as) for f in FINDINGS]