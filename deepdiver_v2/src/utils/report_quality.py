# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
"""Deterministic normalization and delivery gate for report-only artifacts."""

import re
from typing import Dict, List, Tuple


# Only unambiguous double-bracket numeric tuples are protected here. Keep the
# established interpretation of single-bracket citations such as [1, 2].
_NUMERIC_MATH_PATTERN = re.compile(r"\[\[\s*\d+(?:\s*[,;，、]\s*\d+)+\s*\]\]")

# 仅保护明确的中文公文文号；独立 [2014] 仍按既有引用规则处理。
# 中文六角括号等不会被 [数字] 引用规则匹配，不在此处扩展或改写。
_DOCUMENT_NUMBER_PATTERN = re.compile(
    r"[\u4e00-\u9fff]{2,20}[ \t]*\[(?:19|20)\d{2}\][ \t]*\d{1,6}[ \t]*号"
)


def protect_document_numbers(content: str) -> Tuple[str, Dict[str, str]]:
    """Preserve exact document identifiers during source citation remapping."""
    prefix = "__REPORT_DOCUMENT_NUMBER_"
    while prefix in content:
        prefix = "_" + prefix
    fragments = {}

    def protect(match: re.Match) -> str:
        token = f"{prefix}{len(fragments)}__"
        fragments[token] = match.group(0)
        return token

    return _DOCUMENT_NUMBER_PATTERN.sub(protect, content), fragments


def restore_document_numbers(content: str, fragments: Dict[str, str]) -> str:
    """Restore identifiers before saving Markdown, summarizing or rendering PDF."""
    for token, fragment in fragments.items():
        content = content.replace(token, fragment)
    return content


def protect_numeric_math(content: str) -> Tuple[str, Dict[str, str]]:
    """Hide numeric tuples for one citation pass; restore before any output."""
    prefix = "__REPORT_NUMERIC_MATH_"
    while prefix in content:
        prefix = "_" + prefix
    fragments = {}

    def protect(match: re.Match) -> str:
        token = f"{prefix}{len(fragments)}__"
        fragments[token] = match.group(0)
        return token

    return _NUMERIC_MATH_PATTERN.sub(protect, content), fragments


def restore_numeric_math(content: str, fragments: Dict[str, str]) -> str:
    """Restore exact original spelling, spacing and delimiters."""
    for token, fragment in fragments.items():
        content = content.replace(token, fragment)
    return content


_INTERNAL_MARKER_PATTERNS = (
    re.compile(r"\[\s*unused\s*\d+\s*\]", re.IGNORECASE),
    re.compile(r"\[\s*citation[^\]]*\]", re.IGNORECASE),
    re.compile(r"\[\s*webp(?:aeg|age)[^\]]*\]", re.IGNORECASE),
    re.compile(r"\[\s*web\s*\d+[^\]]*\](?!\()", re.IGNORECASE),
)


def normalize_report_artifacts(content: str, *, source_citations: bool = False) -> str:
    """Normalize markers; opt into webN conversion only before source remapping."""
    if not content:
        return content

    content, numeric_math = protect_numeric_math(content)

    normalized = re.sub(
        r"\[\s*unused\s*\d+\s*\]",
        "",
        content,
        flags=re.IGNORECASE,
    )

    def normalize_named_citation(match: re.Match) -> str:
        numbers = re.findall(r"\d+", match.group(1))
        return "".join(f"[{number}]" for number in numbers)

    normalized = re.sub(
        r"\[\s*citation\s*[:：]?[\s]*([\d\s,;，、]+)\]",
        normalize_named_citation,
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\[\s*webp(?:aeg|age)\s*(\d+)\s+begin\s*\]",
        r"[\1]",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\[\s*webp(?:aeg|age)\s*\d+\s+end\s*\]",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\[\s*webp(?:aeg|age)\s*(\d+)\s*\]",
        r"[\1]",
        normalized,
        flags=re.IGNORECASE,
    )

    # webN is a source ID, not a final bibliography number. Only raw chapters
    # may convert it; the final delivery gate must report it instead. Preserve
    # ordinary Markdown link labels such as [web36](https://example.com).
    if source_citations:
        # Repeated prefixes are unambiguous only in raw source-citation mode.
        # Leave Markdown links and malformed/mixed labels for the existing gate.
        normalized = re.sub(
            r"\[\s*(webp(?:aeg|age)\s*\d+(?:\s*[,;，、]\s*webp(?:aeg|age)\s*\d+)+)\s*\](?!\()",
            normalize_named_citation, normalized, flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\[\s*web\s*(\d+)\s*\](?!\()",
            r"[\1]",
            normalized,
            flags=re.IGNORECASE,
        )

    def split_numeric_citations(match: re.Match) -> str:
        return "".join(f"[{number}]" for number in re.findall(r"\d+", match.group(1)))

    normalized = re.sub(
        r"\[\s*(\d+(?:\s*[,;，、]\s*\d+)+)\s*\]",
        split_numeric_citations,
        normalized,
    )
    return restore_numeric_math(normalized, numeric_math)


def find_internal_report_markers(content: str) -> List[str]:
    """Return unique unsupported markers remaining after normalization."""
    if not content:
        return []
    markers = []
    for pattern in _INTERNAL_MARKER_PATTERNS:
        markers.extend(match.group(0) for match in pattern.finditer(content))
    return list(dict.fromkeys(markers))


def normalize_and_validate_report(
    content: str, *, source_citations: bool = False
) -> Tuple[str, List[str]]:
    """Normalize deterministic artifacts and report only residual marker errors.

    Structural and presentation checks intentionally remain outside this P0
    delivery gate so valid GLM/Pangu reports keep their existing success path.
    """
    normalized = normalize_report_artifacts(content, source_citations=source_citations)
    return normalized, find_internal_report_markers(normalized)
