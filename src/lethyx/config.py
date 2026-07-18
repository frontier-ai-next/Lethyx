"""
Unified runtime configuration for the Lethyx manager.

Loads from default_policy.yaml ``runtime:`` section, environment variables,
and CLI overrides.  Every tuning knob (budget, iteration limits, quality
threshold, human checkpoint stages) lives here.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

_POLICY_PATH = Path(__file__).parent / "policies" / "default_policy.yaml"

DEFAULT_DASHBOARD_WORKSPACE_ID = "world_prediction"
DEFAULT_DASHBOARD_TASK_TEXT = "Improve and validate this workspace."
DEFAULT_DASHBOARD_TIMEOUT_HOURS = 24.0
DEFAULT_DASHBOARD_QUALITY_THRESHOLD = 0.95

LETHYX_TOOL_NAMES = [
    "get_gmas_context",
    "search_gmas_knowledge",
    "list_workspace_files",
    "read_workspace_file",
    "run_workspace_command",
    "commit_workspace_changes",
    "get_LETHYX_memory",
    "save_LETHYX_memory",
    "record_workspace_event",
    "update_workspace_seed",
]

LLM_EVAL_TASK_PREVIEW_LIMIT = 1000
LLM_EVAL_ARTIFACT_CONTENT_LIMIT = 5000
LLM_EVAL_ARTIFACT_PREVIEW_LIMIT = 1000
LLM_EVAL_AGENT_OUTPUT_LIMIT = 2000
LLM_EVAL_AGENT_OUTPUT_PREVIEW_LIMIT = 500

LETHYX_BRIDGE_TEXT_PREVIEW_LIMIT = 1000
MANAGER_PROGRESS_TASK_PREVIEW_LIMIT = 8000


class LethyxRuntimeConfig(BaseModel):
    """All runtime knobs for a manager session."""

    # ── Budget ──────────────────────────────────────────────────────────
    max_budget_usd: float | None = None
    max_iterations: int | None = None
    max_duration_seconds: float | None = None

    # ── Quality gate ────────────────────────────────────────────────────
    quality_completion_threshold: float = 0.85
    min_article_word_count: int = 1500
    required_artifact_types: list[str] = Field(default_factory=lambda: ["report"])

    # ── Self-improvement ────────────────────────────────────────────────
    self_improve_after_stalled_iterations: int = 2
    self_improve_max_total_iterations: int = 50

    # ── Human checkpoints ───────────────────────────────────────────────
    human_review_stages: list[str] = Field(
        default_factory=lambda: ["outline_approved", "final_draft"],
    )
    human_review_timeout_seconds: float = 0
    auto_retrieve_gmas_context: bool = False

    # ── Instance retention ───────────────────────────────────────────────
    instance_cleanup_enabled: bool = True
    keep_recent_runs_per_instance: int = 2
    keep_recent_snapshots_per_instance: int = 1
    keep_recent_reports_per_instance: int = 4
    keep_latest_detached_instances: int = 1

    # ── Heartbeat ───────────────────────────────────────────────────────
    heartbeat_interval_seconds: float = 30.0


def _load_runtime_section_from_yaml(path: Path) -> dict[str, Any]:
    """Read the ``runtime:`` key from a YAML policy file."""
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return dict(data.get("runtime") or {}) if isinstance(data, dict) else {}
    except Exception as exc:
        log.debug("Failed to read runtime config from %s: %s", path, exc)
        return {}


def load_runtime_config(
    policy_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> LethyxRuntimeConfig:
    """Build an ``LethyxRuntimeConfig`` by layering YAML defaults + overrides.

    Args:
        policy_path: Path to the policy YAML (defaults to ``Lethyx/policies/default_policy.yaml``).
        overrides: Dict of field overrides (e.g. from CLI flags).  ``None`` values are skipped.

    Returns:
        Fully resolved runtime config.
    """
    yaml_values = _load_runtime_section_from_yaml(policy_path or _POLICY_PATH)

    merged = dict(yaml_values)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value

    return LethyxRuntimeConfig(
        **{k: v for k, v in merged.items() if k in LethyxRuntimeConfig.model_fields}
    )
