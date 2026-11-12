# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
"""Pure validation helpers for managed report chapter and output paths."""

import re
from pathlib import Path
from typing import Any, List, Tuple


_CANONICAL_PART_RE = re.compile(r"^(?:\./)?report/part_(\d+)\.md$")
_MISFORMATTED_PART_RE = re.compile(
    r"^(?:\./)?report/part_(\d+)[._]\d+.*\.md$"
)


def normalize_report_path_text(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def suggested_canonical_part_path(file_path: Any) -> str:
    """Return a safe suggestion while leaving canonical paths unchanged."""
    normalized = normalize_report_path_text(file_path)
    match = _MISFORMATTED_PART_RE.fullmatch(normalized)
    if not match:
        return normalized
    return f"./report/part_{match.group(1)}.md"


def validate_section_file_sequence(section_files: Any) -> List[Tuple[int, str]]:
    """Require canonical, ordered, unique and contiguous part_N.md inputs."""
    if not isinstance(section_files, list) or not section_files:
        raise ValueError("section_files 必须是非空列表")

    validated: List[Tuple[int, str]] = []
    for item in section_files:
        raw_path = item.get("file_path") if isinstance(item, dict) else item
        normalized = normalize_report_path_text(raw_path)
        match = _CANONICAL_PART_RE.fullmatch(normalized)
        if not match:
            suggestion = suggested_canonical_part_path(normalized)
            suffix = f"，建议使用 {suggestion}" if suggestion != normalized else ""
            raise ValueError(f"非规范章节路径: {raw_path}{suffix}")
        validated.append((int(match.group(1)), f"./report/part_{match.group(1)}.md"))

    indices = [index for index, _ in validated]
    duplicates = sorted(index for index in set(indices) if indices.count(index) > 1)
    if duplicates:
        raise ValueError(f"section_files 包含重复章节: {duplicates}")
    expected = list(range(1, max(indices) + 1))
    if sorted(indices) != expected:
        missing = sorted(set(expected) - set(indices))
        raise ValueError(f"section_files 章节编号不连续，缺失章节: {missing}")
    if indices != expected:
        raise ValueError(f"section_files 必须按章节顺序传入，期望 {expected}，实际 {indices}")
    return validated


def resolve_final_report_path(workspace_path: Any, final_file_path: Any) -> Path:
    """Resolve the one managed final-report output inside the task workspace."""
    workspace = Path(workspace_path)
    expected = workspace / "report" / "final_report.md"
    requested_text = normalize_report_path_text(final_file_path)
    if requested_text in {"report/final_report.md", "./report/final_report.md"}:
        return expected

    requested = Path(final_file_path)
    if requested.is_absolute() and requested.resolve() == expected.resolve():
        return expected
    raise ValueError(
        "final_file_path 仅允许 ./report/final_report.md，"
        f"实际收到: {final_file_path}"
    )
