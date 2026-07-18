"""Host-agent dedup helpers for memory scenario harness."""

from pathlib import Path
from typing import Any

from lethyx.evals.memory_scenarios.assertions import assert_single_always_loaded_block


def run_host_agent_dedup_check(
    repo: Path,
    workspace_id: str,
    task: dict[str, Any],
) -> list[str]:
    del repo, workspace_id
    errors: list[str] = []
    overlays = task.get("context_overlays") or {}
    if not overlays.get("prevent_host_auto_core_overlay"):
        errors.append("host auto core overlay guard missing")
    task_input = str(task.get("input") or "")
    errors.extend(assert_single_always_loaded_block(task_input))
    return errors
