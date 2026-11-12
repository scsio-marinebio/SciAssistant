# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
"""Narrow late-stage convergence guard for PlannerAgent.

The guard is deliberately inactive during ordinary planning and research.  It
only applies to long-form tasks after information-seeker assignment has already
completed, preventing optional inspection/research calls from consuming the
last four Planner iterations before Writer delivery.
"""

from typing import Optional, Set, Tuple


_COMPLETION_TOOLS = {
    "planner_subjective_task_done",
    "planner_objective_task_done",
    "writer_subjective_task_done",
}
_WRITER_TOOL = "assign_subjective_task_to_writer"


def get_planner_convergence_policy(
    planner_mode: str,
    total_iterations: int,
    current_iteration: int,
    has_assigned_info_tasks: bool,
    writer_invoked: bool,
) -> Tuple[Optional[str], Optional[Set[str]]]:
    """Return a late-stage instruction and allowlist, or ``(None, None)``.

    No restriction is introduced until information gathering has succeeded and
    at most four iterations remain.  This keeps the existing planning flow
    unchanged while budget is healthy.
    """
    if planner_mode not in {"writing", "auto"} or not has_assigned_info_tasks:
        return None, None

    total = max(1, int(total_iterations))
    current = max(1, int(current_iteration))
    remaining = max(0, total - current + 1)
    convergence_window = min(4, max(2, total // 4))
    if remaining > convergence_window:
        return None, None

    if writer_invoked:
        return (
            f"PLANNER CONVERGENCE: iteration {current}/{total}; {remaining} iteration(s) remain. "
            "Information gathering and Writer execution are complete. Do not research, inspect the "
            "workspace, recreate the plan, or invoke Writer again. Call the appropriate completion "
            "tool now. /no_think",
            set(_COMPLETION_TOOLS),
        )

    allowed_tools = {_WRITER_TOOL}
    if planner_mode == "auto":
        allowed_tools |= set(_COMPLETION_TOOLS)
    return (
        f"PLANNER CONVERGENCE: iteration {current}/{total}; {remaining} iteration(s) remain. "
        "Information gathering is complete. Do not perform more research, workspace inspection, "
        "or planning-file edits. Invoke assign_subjective_task_to_writer now using the collected "
        "key_files, or use an appropriate completion tool if no long-form report is required. "
        "/no_think",
        allowed_tools,
    )
