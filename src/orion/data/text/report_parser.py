"""
Radiology Report Parser

WHY IT EXISTS
-------------

Radiology reports contain valuable weak-label information for the
report-only studies.

Reports are free text, so the first step is to separate clinically
meaningful sections before passing text to downstream components such as
the weak-label extractor or text encoder.

The parser produces four stable sections:

    clinical_indication
    technique
    findings
    impression

IMPORTANT
---------

This is a heuristic parser, not a clinical NLP system.

It deliberately does not attempt to infer diagnoses. Its job is only to
identify section boundaries and preserve the text belonging to each section.

The parser is designed for messy real-world radiology formatting:

    FINDINGS:
    Findings:
    FINDINGS
    Findings
    IMPRESSION:
    CONCLUSION:
    OPINION:
    CLINICAL HISTORY:
    HISTORY:
    PROCEDURE:
    TECHNIQUE:

Unknown sections such as COMPARISON or COMMENT are treated as boundaries
rather than silently becoming part of the preceding clinical section.
"""

from __future__ import annotations

import re


__all__ = [
    "extract_sections",
]


# ---------------------------------------------------------------------------
# Canonical output sections
# ---------------------------------------------------------------------------

_SECTION_NAMES = (
    "clinical_indication",
    "technique",
    "findings",
    "impression",
)


# ---------------------------------------------------------------------------
# Header vocabulary
# ---------------------------------------------------------------------------
#
# The order matters for aliases that overlap.
#
# We keep this vocabulary conservative. A line is considered a section
# header only when the complete line is a known header, optionally followed
# by a colon or dash and inline section content.
# ---------------------------------------------------------------------------

_SECTION_HEADERS: dict[str, tuple[str, ...]] = {
    "clinical_indication": (
        "clinical indication",
        "clinical indications",
        "indication",
        "indications",
        "clinical history",
        "history",
        "history of present illness",
        "reason for exam",
        "reason for examination",
        "reason for study",
        "clinical information",
    ),

    "technique": (
        "technique",
        "procedure",
        "procedures",
        "examination",
        "exam",
        "protocol",
    ),

    "findings": (
        "findings",
        "finding",
        "observations",
        "observation",
        "description",
    ),

    "impression": (
        "impression",
        "impressions",
        "conclusion",
        "conclusions",
        "opinion",
        "summary",
    ),
}


# ---------------------------------------------------------------------------
# Headers which should terminate a known section without becoming one of
# our returned sections.
#
# This prevents text such as:
#
#     FINDINGS:
#     ...
#     COMPARISON:
#     ...
#
# from incorrectly assigning COMPARISON text to FINDINGS.
# ---------------------------------------------------------------------------

_BOUNDARY_ONLY_HEADERS = (
    "comparison",
    "comparisons",
    "comment",
    "comments",
    "recommendation",
    "recommendations",
    "addendum",
    "addenda",
    "discussion",
)


# ---------------------------------------------------------------------------
# Header regex
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(
    r"""
    ^
    (?P<header>
        [A-Za-z][A-Za-z0-9 /_-]*?
    )
    \s*
    (?:
        :
        |
        -
    )?
    \s*
    (?P<content>.*?)
    $
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_sections() -> dict[str, str]:
    """Return a fresh empty section dictionary."""

    return {
        section: ""
        for section in _SECTION_NAMES
    }


def _normalize_header(header: str) -> str:
    """
    Normalize a candidate section header.

    Multiple spaces, underscores, and hyphens are normalized so that
    formatting variations such as:

        CLINICAL_HISTORY
        Clinical-History
        Clinical History

    can be handled consistently.
    """

    normalized = header.strip().lower()

    normalized = re.sub(
        r"[_-]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def _resolve_header(
    header: str,
) -> str | None:
    """
    Map a normalized header to one canonical output section.

    Returns None for ordinary content lines and boundary-only headers.
    """

    normalized = _normalize_header(
        header
    )

    if not normalized:
        return None

    for section, aliases in _SECTION_HEADERS.items():
        if normalized in aliases:
            return section

    return None


def _is_boundary_header(
    header: str,
) -> bool:
    """Return True when a header should terminate the current section."""

    normalized = _normalize_header(
        header
    )

    return normalized in _BOUNDARY_ONLY_HEADERS


def _looks_like_header(
    line: str,
) -> bool:
    """
    Determine whether a line plausibly represents a section header.

    A known header is always accepted.

    Unknown headers are accepted as boundaries only when they have an
    explicit colon/dash or are written in a conventional short-header form.
    """

    match = _HEADER_RE.match(line)

    if match is None:
        return False

    header = match.group(
        "header"
    ).strip()

    if not header:
        return False

    if _resolve_header(header) is not None:
        return True

    if _is_boundary_header(header):
        return True

    # Avoid interpreting ordinary prose as a header.
    #
    # Unknown header-like lines should be short and visually structured.
    if len(header) > 40:
        return False

    has_explicit_separator = bool(
        re.match(
            r"^[^:]+[:\-]\s*",
            line,
        )
    )

    if has_explicit_separator:
        return True

    # Accept conventional uppercase section labels such as:
    #
    #     COMPARISON
    #
    # but do not classify arbitrary title-case prose as a header.
    return header.isupper() and len(
        header.split()
    ) <= 5


def _parse_header_line(
    line: str,
) -> tuple[str | None, str]:
    """
    Parse one possible header line.

    Returns
    -------
    tuple[str | None, str]
        Canonical section name and inline content.

    Examples
    --------
    "FINDINGS:" ->
        ("findings", "")

    "FINDINGS: Small joint effusion." ->
        ("findings", "Small joint effusion.")

    "IMPRESSION" ->
        ("impression", "")
    """

    match = _HEADER_RE.match(line)

    if match is None:
        return None, ""

    raw_header = match.group(
        "header"
    ).strip()

    inline_content = match.group(
        "content"
    ).strip()

    section = _resolve_header(
        raw_header
    )

    if section is not None:
        return section, inline_content

    if _is_boundary_header(
        raw_header
    ):
        return "__boundary__", ""

    # Unknown header-like text is a boundary but is not retained in one of
    # the canonical sections.
    if _looks_like_header(line):
        return "__boundary__", ""

    return None, ""


def _clean_text(
    parts: list[str],
) -> str:
    """
    Join section fragments while preserving readable sentence spacing.
    """

    if not parts:
        return ""

    text = " ".join(
        part.strip()
        for part in parts
        if part and part.strip()
    )

    # Normalize whitespace introduced by line joining.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_sections(
    report_text: str,
) -> dict[str, str]:
    """
    Parse a radiology report into canonical sections.

    Parameters
    ----------
    report_text:
        Raw radiology report text.

    Returns
    -------
    dict[str, str]
        Dictionary containing exactly:

            clinical_indication
            technique
            findings
            impression

    Notes
    -----

    1. Missing sections remain empty strings.

    2. Section names are normalized to lowercase canonical names.

    3. Header aliases such as FINDINGS, OBSERVATIONS, IMPRESSION,
       CONCLUSION, HISTORY, and TECHNIQUE are supported.

    4. Inline content after a header is retained.

    5. Unknown section headers terminate the current known section so that
       unrelated content is not silently attributed to the previous section.

    6. No diagnosis is inferred here.
    """

    sections = _empty_sections()

    if not isinstance(
        report_text,
        str,
    ):
        return sections

    text = report_text.strip()

    if not text:
        return sections

    lines = text.splitlines()

    current_section: str | None = None

    buffers: dict[str, list[str]] = {
        section: []
        for section in _SECTION_NAMES
    }

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            continue

        section, inline_content = (
            _parse_header_line(line)
        )

        # ---------------------------------------------------------------
        # Known section header
        # ---------------------------------------------------------------

        if section in _SECTION_NAMES:

            current_section = section

            if inline_content:
                buffers[
                    current_section
                ].append(
                    inline_content
                )

            continue

        # ---------------------------------------------------------------
        # Boundary-only / unknown header
        # ---------------------------------------------------------------

        if section == "__boundary__":
            current_section = None
            continue

        # ---------------------------------------------------------------
        # Ordinary content
        # ---------------------------------------------------------------

        if current_section is not None:
            buffers[
                current_section
            ].append(line)

    # -------------------------------------------------------------------
    # Final cleanup
    # -------------------------------------------------------------------

    for section in _SECTION_NAMES:
        sections[section] = _clean_text(
            buffers[section]
        )

    return sections
