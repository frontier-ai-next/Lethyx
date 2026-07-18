"""Active lifecycle manager: proposals and reports, not self-promotion."""

from pathlib import Path

from lethyx.memory.kernel.value import load_memory_value_stats
from lethyx.memory.lifecycle.models import LifecycleProposal
from lethyx.memory.lifecycle.policies import propose_lifecycle_action
from lethyx.memory.proactive.bkb import load_bkb_rules


def evaluate_lifecycle(
    repo_root: Path, drive_root: Path, workspace_id: str = ""
) -> list[LifecycleProposal]:
    bkb_paths = [Path(repo_root) / ".Lethyx" / "memory" / "core" / "bkb.yaml"]
    if workspace_id:
        bkb_paths.append(
            Path(repo_root) / "workspaces" / workspace_id / ".memory" / "core" / "bkb.yaml"
        )
    stats = load_memory_value_stats(drive_root)
    proposals: list[LifecycleProposal] = []
    for path in bkb_paths:
        for rule in load_bkb_rules(path):
            proposals.append(propose_lifecycle_action(rule, stats.get(rule.id)))
    return proposals


def apply_safe_lifecycle_state(proposals: list[LifecycleProposal]) -> dict[str, int]:
    # Safe apply means reporting/downrank hints only; never promotion/deletion.
    return {
        "safe_state_updates": sum(1 for p in proposals if p.safe_to_apply),
        "promotions": 0,
        "deletions": 0,
    }
