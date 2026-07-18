"""
Competency ledger - tracks manager capability gaps and signals.

Detects repeated failure patterns that may indicate:
- Manager-level capability gaps (vs workspace issues)
- Need for self-improvement
- Missing tools or retrieval capabilities
"""

import logging
import time
from collections import defaultdict
from typing import Any

from lethyx.memory.models import (
    CompetencyGapRecord,
    CapabilitySignal,
    GapSeverity,
    GapStatus,
    SignalCategory,
    generate_gap_id,
    generate_signal_id,
)
from lethyx.memory.store import MemoryStore

log = logging.getLogger(__name__)


# =============================================================================
# Public API
# =============================================================================


def record_competency_signal(
    store: MemoryStore,
    category: SignalCategory,
    capability_area: str,
    strength: float,
    evidence_summary: str,
    task_id: str,
    workspace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CapabilitySignal:
    """Record a capability signal (positive or negative).

    Args:
        store: Memory store
        category: Signal category
        capability_area: Area this relates to (e.g., "gmas_knowledge", "retrieval")
        strength: Signal strength (-1.0 to 1.0, negative = problem)
        evidence_summary: What happened
        task_id: Associated task
        workspace_id: Associated workspace if any
        metadata: Additional context

    Returns:
        The created signal
    """
    signal = CapabilitySignal(
        id=generate_signal_id(),
        category=category,
        capability_area=capability_area,
        strength=strength,
        evidence_summary=evidence_summary,
        task_id=task_id,
        workspace_id=workspace_id,
        metadata=metadata or {},
    )

    store.add_signal(signal)

    # Auto-check if we should open/update a gap
    if strength < 0:
        _check_and_update_gap(store, signal)

    return signal


def open_competency_gap(
    store: MemoryStore,
    capability_area: str,
    severity: GapSeverity,
    description: str,
    evidence_signals: list[str] | None = None,
    suggested_actions: list[str] | None = None,
    suspected_root_cause: str | None = None,
    is_workspace_level: bool = False,
) -> CompetencyGapRecord:
    """Open a new competency gap.

    Args:
        store: Memory store
        capability_area: Area with the gap
        severity: Gap severity
        description: What the gap is
        evidence_signals: IDs of signals that evidence this gap
        suggested_actions: What might address it
        suspected_root_cause: What we think causes it
        is_workspace_level: True if problem is in workspace, not manager

    Returns:
        The created gap record
    """
    gap = CompetencyGapRecord(
        id=generate_gap_id(),
        capability_area=capability_area,
        severity=severity,
        description=description,
        evidence_signals=evidence_signals or [],
        suggested_actions=suggested_actions or [],
        suspected_root_cause=suspected_root_cause,
        is_workspace_level=is_workspace_level,
    )

    store.add_gap(gap)
    log.warning(f"Opened competency gap {gap.id}: {capability_area} - {description}")
    return gap


def update_competency_gap(
    store: MemoryStore,
    gap_id: str,
    new_evidence: str | None = None,
    status_override: GapStatus | None = None,
    severity_override: GapSeverity | None = None,
    additional_actions: list[str] | None = None,
) -> bool:
    """Update an existing competency gap with new information.

    Args:
        store: Memory store
        gap_id: Gap to update
        new_evidence: New evidence to add
        status_override: New status if set
        severity_override: New severity if set
        additional_actions: More suggested actions

    Returns:
        True if gap was found and updated
    """
    gap = store.get_gap(gap_id)
    if gap is None:
        return False

    # Touch to update last_seen_at
    gap.touch()

    if new_evidence and new_evidence not in gap.description:
        gap.description = f"{gap.description}\nRecent: {new_evidence}"

    if status_override:
        gap.status = status_override

    if severity_override:
        gap.severity = severity_override

    if additional_actions:
        gap.suggested_actions.extend(additional_actions)

    return True


def get_active_gaps(
    store: MemoryStore,
    capability_area: str | None = None,
    min_severity: GapSeverity | None = None,
) -> list[CompetencyGapRecord]:
    """Get all active competency gaps.

    Args:
        store: Memory store
        capability_area: Filter by area
        min_severity: Minimum severity to include

    Returns:
        List of active gaps, sorted by severity and recency
    """
    gaps = store.get_active_gaps(capability_area)

    if min_severity:
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        min_level = severity_order.get(min_severity, 0)
        gaps = [g for g in gaps if severity_order.get(g.severity, 0) >= min_level]

    return gaps


def check_capability_deficit(
    store: MemoryStore,
    capability_area: str,
    lookback_seconds: float = 86400,  # 24 hours
    min_signals: int = 3,
) -> tuple[bool, list[CapabilitySignal]]:
    """Check if there's a capability deficit in an area.

    Args:
        store: Memory store
        capability_area: Area to check
        lookback_seconds: How far back to look
        min_signals: Minimum negative signals to indicate deficit

    Returns:
        (has_deficit, relevant_signals)
    """

    cutoff = time.time() - lookback_seconds

    # Get recent signals for this area
    all_signals = store.get_signals(capability_area=capability_area, limit=200)
    recent_negative = [s for s in all_signals if s.is_negative and s.timestamp > cutoff]

    has_deficit = len(recent_negative) >= min_signals
    return has_deficit, recent_negative


# =============================================================================
# Internal Helpers
# =============================================================================


def _check_and_update_gap(store: MemoryStore, signal: CapabilitySignal) -> None:
    """Check if a signal should trigger a gap update."""

    # Look for existing gaps in this area
    existing_gaps = store.get_active_gaps(capability_area=signal.capability_area)

    # Check cooldown - don't update same gap too frequently
    cooldown = store.config.gap_cooldown_seconds
    cutoff = time.time() - cooldown

    for gap in existing_gaps:
        if gap.last_seen_at > cutoff:
            # Recent activity on this gap - update it
            gap.touch()
            gap.evidence_signals.append(signal.id)
            return

    # Check if we have enough signals to open a new gap
    recent_signals = store.get_recent_negative_signals(
        seconds=store.config.gap_cooldown_seconds * 3,  # Look back a bit longer
        limit=store.config.gap_threshold_signals + 1,
    )

    area_signals = [s for s in recent_signals if s.capability_area == signal.capability_area]

    if len(area_signals) >= store.config.gap_threshold_signals:
        # Open a new gap
        severity = _infer_severity_from_signals(area_signals)
        description = _describe_signal_cluster(area_signals)

        open_competency_gap(
            store=store,
            capability_area=signal.capability_area,
            severity=severity,
            description=description,
            evidence_signals=[s.id for s in area_signals],
            suggested_actions=_suggest_actions_for_category(
                signal.category, signal.capability_area
            ),
            is_workspace_level=_is_likely_workspace_issue(signal.category, area_signals),
        )


def _infer_severity_from_signals(signals: list[CapabilitySignal]) -> GapSeverity:
    """Infer gap severity from signal patterns."""
    avg_strength = sum(s.strength for s in signals) / len(signals)

    # Count occurrences
    occurrence_counts = defaultdict(int)
    for s in signals:
        occurrence_counts[s.category] += 1

    max_occurrences = max(occurrence_counts.values())

    # Severity based on strength and frequency
    if avg_strength < -0.8 or max_occurrences >= 5:
        return GapSeverity.CRITICAL
    elif avg_strength < -0.5 or max_occurrences >= 3:
        return GapSeverity.HIGH
    elif avg_strength < -0.3:
        return GapSeverity.MEDIUM
    else:
        return GapSeverity.LOW


def _describe_signal_cluster(signals: list[CapabilitySignal]) -> str:
    """Create a human-readable description of a signal cluster."""
    if not signals:
        return "Unknown issue"

    # Most common category
    categories = [s.category for s in signals]
    top_category = max(set(categories), key=categories.count)

    # Get unique evidence snippets
    evidence_snippets = list({s.evidence_summary[:100] for s in signals})[:3]

    descriptions = {
        SignalCategory.NO_PROGRESS_ITERATIONS: "Repeated workspace iterations without progress",
        SignalCategory.RETRIEVAL_MISSES: "Repeated failures to retrieve relevant information",
        SignalCategory.REPEATED_FAILURE_MODE: "Same failure pattern occurring across runs",
        SignalCategory.HUMAN_FEEDBACK: "Negative feedback from human evaluators",
        SignalCategory.HIGH_COST_NO_GAIN: "High resource consumption without quality improvement",
        SignalCategory.MISSING_CAPABILITY: "Missing required capability or tool",
    }

    base = descriptions.get(top_category, "Repeated capability issues")

    if evidence_snippets:
        base += f": {'; '.join(evidence_snippets)}"

    return base


def _suggest_actions_for_category(category: SignalCategory, capability_area: str) -> list[str]:
    """Suggest actions for a given signal category."""
    suggestions = {
        SignalCategory.NO_PROGRESS_ITERATIONS: [
            "Review workspace strategy and approach",
            "Consider switching to a different workspace",
            "Check if task constraints are achievable",
        ],
        SignalCategory.RETRIEVAL_MISSES: [
            "Improve retrieval query planning",
            "Expand retrieval index coverage",
            "Review search term selection",
        ],
        SignalCategory.REPEATED_FAILURE_MODE: [
            "Analyze failure signature for root cause",
            "Consider if this is a workspace vs manager issue",
            "Review if GMAS patterns are being correctly applied",
        ],
        SignalCategory.HUMAN_FEEDBACK: [
            "Review specific feedback points",
            "Consider adjusting strategy based on input",
            "Verify understanding of task requirements",
        ],
        SignalCategory.HIGH_COST_NO_GAIN: [
            "Review token usage and efficiency",
            "Consider simpler approaches",
            "Add early stopping criteria",
        ],
        SignalCategory.MISSING_CAPABILITY: [
            f"Add capability: {capability_area}",
            "Consider if external tool can help",
            "Evaluate if this is a core manager gap",
        ],
    }

    return suggestions.get(category, ["Investigate the underlying issue"])


def should_trigger_self_improvement(
    store: MemoryStore,
    *,
    lookback_seconds: float = 7200,
    min_negative_signals: int = 3,
    min_open_gaps: int = 1,
) -> tuple[bool, str]:
    """Decide whether to trigger a self-improvement cycle.

    This acts as the automatic trigger described in the plan: when enough
    negative competency signals accumulate, the system should switch from
    workspace-patching to self-patching.

    Returns:
        (should_trigger, reason)
    """
    recent_negatives = store.get_recent_negative_signals(
        seconds=lookback_seconds,
        limit=100,
    )
    active_gaps = store.get_active_gaps()
    critical_gaps = [g for g in active_gaps if g.severity in ("critical", "high")]

    if len(critical_gaps) >= 1:
        return True, (
            f"{len(critical_gaps)} critical/high gap(s): "
            f"{critical_gaps[0].capability_area} - {critical_gaps[0].description[:80]}"
        )

    if len(recent_negatives) >= min_negative_signals and len(active_gaps) >= min_open_gaps:
        areas = {s.capability_area for s in recent_negatives}
        return True, (
            f"{len(recent_negatives)} negative signals in {len(areas)} area(s) "
            f"with {len(active_gaps)} open gap(s)"
        )

    return False, "Competency ledger within normal thresholds"


def _is_likely_workspace_issue(category: SignalCategory, signals: list[CapabilitySignal]) -> bool:
    """Determine if a signal cluster likely indicates a workspace issue vs manager issue."""

    # These categories are more likely to be workspace-level
    workspace_categories = {
        SignalCategory.NO_PROGRESS_ITERATIONS,  # Often workspace-specific
        SignalCategory.HIGH_COST_NO_GAIN,  # Often workspace configuration
    }

    # If all signals are from the same workspace, likely workspace issue
    workspace_ids = [s.workspace_id for s in signals if s.workspace_id]
    if len(set(workspace_ids)) == 1:
        return True

    # Manager-level categories
    if category in {
        SignalCategory.RETRIEVAL_MISSES,
        SignalCategory.MISSING_CAPABILITY,
    }:
        return False

    # Default to assuming workspace issue for these categories
    return category in workspace_categories
