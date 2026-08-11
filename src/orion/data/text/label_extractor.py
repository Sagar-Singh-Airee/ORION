
"""
Weak-label extraction from free-text radiology reports.

WHY IT EXISTS
-------------

The dataset contains approximately 58 expert-labeled studies and 4,900+
report-only studies.

This module converts report text into weak supervision for the report-only
studies.

Each finding receives a ternary label:

    1   = positive evidence
    0   = explicit negative evidence
   -1   = not mentioned, uncertain, or conflicting evidence

IMPORTANT
---------

- -1 must be treated as missing/unknown by the training loss.
- -1 must NOT be converted to 0.
- This module is a labeling function, not a clinical diagnosis system.
- Evidence is retained so extracted labels can be audited.
- The extractor assumes English text. Translation should occur upstream.

DESIGN
------

The extractor uses:

    finding trigger
        ↓
    local context
        ↓
    negation detection
        ↓
    uncertainty detection
        ↓
    normal/anatomical-status detection
        ↓
    evidence aggregation
        ↓
    ternary weak label

The implementation deliberately avoids a large unrestricted character window.
Negation and uncertainty are evaluated using nearby tokens and sentence-local
boundaries so that one finding does not accidentally inherit the status of a
different finding elsewhere in the sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


__all__ = [
    "FINDINGS",
    "FINDING_PATTERNS",
    "ExtractionResult",
    "extract_labels",
    "extract_labels_batch",
    "labels_to_vector",
]


# ---------------------------------------------------------------------------
# Canonical finding order
# ---------------------------------------------------------------------------

FINDINGS = [
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
]


# ---------------------------------------------------------------------------
# Finding triggers
# ---------------------------------------------------------------------------
#
# These are intentionally conservative.
#
# A trigger means:
#
#     "This sentence contains language relevant to this finding."
#
# It does NOT by itself mean the finding is positive.
#
# Negation, uncertainty, and normal-anatomy handling happen later.
# ---------------------------------------------------------------------------

FINDING_PATTERNS: dict[str, list[str]] = {
    "acl": [
        r"\bacl\b",
        r"\banterior\s+cruciate\s+ligament\b",
    ],

    "mcl": [
        r"\bmcl\b",
        r"\bmedial\s+collateral\s+ligament\b",
    ],

    "medial_meniscus": [
        r"\bmedial\s+meniscus\b",
        r"\bmedial\s+meniscal\b",
    ],

    "lateral_meniscus": [
        r"\blateral\s+meniscus\b",
        r"\blateral\s+meniscal\b",
    ],

    "medial_oa": [
        r"\bmedial\s+compartment\b.{0,30}"
        r"(?:osteoarthrit|degenerat|joint\s+space\s+narrow)",
        r"\bmedial\b.{0,20}"
        r"\bosteoarthrit",
    ],

    "lateral_oa": [
        r"\blateral\s+compartment\b.{0,30}"
        r"(?:osteoarthrit|degenerat|joint\s+space\s+narrow)",
        r"\blateral\b.{0,20}"
        r"\bosteoarthrit",
    ],

    "pf_oa": [
        r"\bpatellofemoral\b.{0,30}"
        r"(?:osteoarthrit|degenerat|joint\s+space\s+narrow|"
        r"compartment)",
        r"\bpatellofemoral\s+compartment\b",
    ],

    "effusion": [
        r"\bjoint\s+effusion\b",
        r"\bknee\s+effusion\b",
        r"\beffusion\b",
    ],

    "synovitis": [
        r"\bsynovitis\b",
        r"\bsynovial\s+"
        r"(?:thickening|proliferation|hypertrophy)\b",
    ],

    "bakers_cyst": [
        r"\bbaker['’]?s?\s+cyst\b",
        r"\bpopliteal\s+cyst\b",
    ],

    "contusion": [
        r"\bbone\s+bruise\b",
        r"\bbone\s+bruising\b",
        r"\bbone\s+contusion\b",
        r"\bcontusion\b",
    ],

    "fracture": [
        r"\bfracture\b",
        r"\bfx\b",
    ],
}


# ---------------------------------------------------------------------------
# Negation cues
# ---------------------------------------------------------------------------
#
# These cues are deliberately conservative.
#
# "denies" is NOT included because a patient's denial of symptoms is not
# equivalent to a radiologist stating that an imaging finding is absent.
# ---------------------------------------------------------------------------

PRE_NEGATION = [
    r"no",
    r"no\s+evidence\s+of",
    r"no\s+signs?\s+of",
    r"without",
    r"without\s+evidence\s+of",
    r"negative\s+for",
    r"absent",
    r"free\s+of",
    r"lack\s+of",
]

POST_NEGATION = [
    r"not\s+seen",
    r"not\s+identified",
    r"not\s+present",
    r"is\s+not\s+identified",
    r"was\s+not\s+identified",
    r"is\s+absent",
    r"are\s+absent",
]


# ---------------------------------------------------------------------------
# Uncertainty cues
# ---------------------------------------------------------------------------

UNCERTAINTY = [
    r"cannot\s+exclude",
    r"cannot\s+rule\s+out",
    r"cannot\s+be\s+excluded",
    r"possible",
    r"possibly",
    r"probable",
    r"probably",
    r"suspected",
    r"suspicious\s+for",
    r"suggestive\s+of",
    r"may\s+represent",
    r"may\s+reflect",
    r"could\s+represent",
    r"could\s+reflect",
    r"equivocal",
    r"questionable",
    r"differential\s+includes",
    r"not\s+fully\s+excluded",
]


# ---------------------------------------------------------------------------
# Normal / intact anatomy cues
# ---------------------------------------------------------------------------
#
# These are important for findings such as ACL, MCL, and meniscus where a
# report commonly mentions the structure specifically while saying that it
# is intact or preserved.
# ---------------------------------------------------------------------------

NORMAL_CUES = [
    r"\bintact\b",
    r"\bpreserved\b",
    r"\bnormal\b",
    r"\bunremarkable\b",
    r"\bmaintained\b",
    r"\bno\s+tear\b",
    r"\bwithout\s+tear\b",
]


# ---------------------------------------------------------------------------
# Sentence and token handling
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(
    r"(?:\r?\n)+|(?<=[.!?;])\s+"
)

_TOKEN_RE = re.compile(
    r"\b[\w’'-]+\b",
    re.UNICODE,
)

# Maximum number of tokens used when looking for local contextual cues.
#
# This is deliberately much smaller than the old 60-character window.
# A character window can cross unrelated clauses and accidentally transfer
# negation/uncertainty from one finding to another.
_CONTEXT_TOKENS = 8


# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

_COMPILED_FINDINGS: dict[
    str,
    tuple[re.Pattern[str], ...],
] = {
    finding: tuple(
        re.compile(
            pattern,
            re.IGNORECASE,
        )
        for pattern in patterns
    )
    for finding, patterns in FINDING_PATTERNS.items()
}

_COMPILED_PRE_NEGATION = tuple(
    re.compile(
        pattern,
        re.IGNORECASE,
    )
    for pattern in PRE_NEGATION
)

_COMPILED_POST_NEGATION = tuple(
    re.compile(
        pattern,
        re.IGNORECASE,
    )
    for pattern in POST_NEGATION
)

_COMPILED_UNCERTAINTY = tuple(
    re.compile(
        pattern,
        re.IGNORECASE,
    )
    for pattern in UNCERTAINTY
)

_COMPILED_NORMAL = tuple(
    re.compile(
        pattern,
        re.IGNORECASE,
    )
    for pattern in NORMAL_CUES
)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """
    Weak-label extraction result for one report.

    labels:
        finding -> {-1, 0, 1}

    evidence:
        finding -> sentence containing the strongest evidence.

    A finding may be absent from evidence when no trigger was detected.
    """

    labels: dict[str, int] = field(
        default_factory=dict
    )

    evidence: dict[str, str] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Internal evidence representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Evidence:
    """One finding mention detected in one sentence."""

    label: int
    sentence: str
    start: int
    end: int


# ---------------------------------------------------------------------------
# Sentence normalization
# ---------------------------------------------------------------------------


def _split_sentences(
    text: str,
) -> list[str]:
    """
    Split report text into conservative sentence-like units.

    Radiology reports frequently use semicolon-separated findings and
    line-separated observations, so both are treated as boundaries.
    """

    return [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(
            text
        )
        if sentence.strip()
    ]


def _token_positions(
    text: str,
) -> list[tuple[int, int]]:
    """Return character spans for word-like tokens."""

    return [
        match.span()
        for match in _TOKEN_RE.finditer(
            text
        )
    ]


def _token_index_at_or_before(
    positions: list[tuple[int, int]],
    character_position: int,
) -> int:
    """Find the last token beginning before a character position."""

    index = -1

    for i, (start, end) in enumerate(
        positions
    ):
        if start >= character_position:
            break
        index = i

    return index


# ---------------------------------------------------------------------------
# Local context
# ---------------------------------------------------------------------------


def _local_context(
    sentence: str,
    start: int,
    end: int,
) -> tuple[str, str]:
    """
    Return local text before and after a finding mention.

    Context is limited by token count rather than character count.
    """

    positions = _token_positions(
        sentence
    )

    if not positions:
        return "", ""

    match_token_start = (
        _token_index_at_or_before(
            positions,
            start,
        )
    )

    match_token_end = (
        _token_index_at_or_before(
            positions,
            end,
        )
    )

    # A regex match may end before the end of the token containing it.
    if (
        match_token_end >= 0
        and positions[match_token_end][1] < end
    ):
        match_token_end += 1

    if match_token_start < 0:
        before_start = max(
            0,
            start - 80,
        )
    else:
        before_token = max(
            0,
            match_token_start
            - _CONTEXT_TOKENS,
        )
        before_start = positions[
            before_token
        ][0]

    if match_token_end < 0:
        after_end = min(
            len(sentence),
            end + 80,
        )
    else:
        after_token = min(
            len(positions) - 1,
            match_token_end
            + _CONTEXT_TOKENS
            - 1,
        )
        after_end = positions[
            after_token
        ][1]

    return (
        sentence[before_start:start],
        sentence[end:after_end],
    )


# ---------------------------------------------------------------------------
# Cue matching
# ---------------------------------------------------------------------------


def _has_pre_negation(
    text: str,
) -> bool:
    """Return whether text immediately preceding the finding contains negation."""

    return any(
        pattern.search(text)
        for pattern in _COMPILED_PRE_NEGATION
    )


def _has_post_negation(
    text: str,
) -> bool:
    """Return whether text immediately following the finding contains negation."""

    return any(
        pattern.search(text)
        for pattern in _COMPILED_POST_NEGATION
    )


def _has_uncertainty(
    before: str,
    after: str,
) -> bool:
    """Return whether local context contains an uncertainty cue."""

    return any(
        pattern.search(before)
        or pattern.search(after)
        for pattern in _COMPILED_UNCERTAINTY
    )


def _has_normal_cue(
    before: str,
    after: str,
) -> bool:
    """Return whether local context describes normal/intact anatomy."""

    return any(
        pattern.search(before)
        or pattern.search(after)
        for pattern in _COMPILED_NORMAL
    )


# ---------------------------------------------------------------------------
# Mention classification
# ---------------------------------------------------------------------------


def _classify_mention(
    sentence: str,
    start: int,
    end: int,
) -> int:
    """
    Classify one finding mention.

    Priority:

        explicit negation
            ↓
        uncertainty
            ↓
        explicit normal/intact anatomy
            ↓
        positive

    Negation is evaluated before uncertainty because phrases such as
    "cannot exclude no fracture" are unusual and should not allow an
    uncertainty cue to override an explicit negative statement.
    """

    before, after = _local_context(
        sentence,
        start,
        end,
    )

    if _has_pre_negation(
        before
    ):
        return 0

    if _has_post_negation(
        after
    ):
        return 0

    if _has_uncertainty(
        before,
        after,
    ):
        return -1

    if _has_normal_cue(
        before,
        after,
    ):
        return 0

    return 1


# ---------------------------------------------------------------------------
# Evidence aggregation
# ---------------------------------------------------------------------------


def _aggregate_evidence(
    evidence: list[_Evidence],
) -> tuple[int, _Evidence | None]:
    """
    Aggregate all mentions of one finding.

    Rules
    -----

    No evidence:
        -1

    Positive + negative:
        -1

    Positive + uncertain:
        1

    Negative + uncertain:
        0

    Only positive:
        1

    Only negative:
        0

    Only uncertain:
        -1

    Rationale
    ---------

    A positive mention should not automatically erase a contradictory
    explicit negative mention. Contradictory reports are safer as unknown
    weak labels than as confidently positive or negative labels.
    """

    if not evidence:
        return -1, None

    labels = {
        item.label
        for item in evidence
    }

    if 1 in labels and 0 in labels:
        # Contradictory evidence.
        #
        # Choose the first explicit contradiction as the retained evidence
        # so auditing can see what triggered the decision.
        return -1, evidence[0]

    if 1 in labels:
        # Positive evidence is stronger than uncertainty when there is no
        # contradictory explicit negative statement.
        positive = next(
            item
            for item in evidence
            if item.label == 1
        )
        return 1, positive

    if 0 in labels:
        negative = next(
            item
            for item in evidence
            if item.label == 0
        )
        return 0, negative

    # Every mention was uncertain.
    uncertain = evidence[0]

    return -1, uncertain


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------


def extract_labels(
    report_text: str,
) -> ExtractionResult:
    """
    Extract ternary weak labels for all 12 findings.

    Parameters
    ----------
    report_text:
        English radiology report text.

    Returns
    -------
    ExtractionResult
        Always contains all FINDINGS with labels in {-1, 0, 1}.

    Notes
    -----
    -1 means unknown/uncertain/not mentioned.
    It must be treated as missing during training, not as a negative label.
    """

    result = ExtractionResult(
        labels={
            finding: -1
            for finding in FINDINGS
        }
    )

    if not isinstance(
        report_text,
        str,
    ):
        return result

    text = report_text.strip()

    if not text:
        return result

    sentences = _split_sentences(
        text
    )

    for finding in FINDINGS:

        evidence: list[_Evidence] = []

        for sentence in sentences:

            patterns = _COMPILED_FINDINGS[
                finding
            ]

            for pattern in patterns:

                for match in pattern.finditer(
                    sentence
                ):

                    label = _classify_mention(
                        sentence,
                        match.start(),
                        match.end(),
                    )

                    evidence.append(
                        _Evidence(
                            label=label,
                            sentence=sentence.strip(),
                            start=match.start(),
                            end=match.end(),
                        )
                    )

        label, strongest = (
            _aggregate_evidence(
                evidence
            )
        )

        result.labels[
            finding
        ] = label

        if strongest is not None:
            result.evidence[
                finding
            ] = strongest.sentence

    return result


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


def extract_labels_batch(
    report_texts: list[str],
) -> list[ExtractionResult]:
    """
    Extract labels from multiple reports.

    This function intentionally remains a simple deterministic wrapper.
    Model-based batching is unnecessary because this extractor is regex/
    rule-based.
    """

    return [
        extract_labels(text)
        for text in report_texts
    ]


# ---------------------------------------------------------------------------
# Vector conversion
# ---------------------------------------------------------------------------


def labels_to_vector(
    result: ExtractionResult,
    missing_as: int = -1,
) -> list[int]:
    """
    Convert an ExtractionResult to the canonical 12-element label vector.

    The ordering exactly matches FINDINGS.
    """

    if missing_as not in (
        -1,
        0,
        1,
    ):
        raise ValueError(
            "missing_as must be one of -1, 0, or 1"
        )

    return [
        int(
            result.labels.get(
                finding,
                missing_as,
            )
        )
        for finding in FINDINGS
    ]
