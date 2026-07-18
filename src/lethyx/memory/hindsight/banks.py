"""Hindsight bank identifiers and missions."""

from dataclasses import dataclass
from typing import Any


MANAGER_BANK_ID = "ub:manager"
GENERIC_AGENT_BANK_ID = "ub:agent:generic_agent"


MANAGER_RETAIN_MISSION = (
    "Capture durable technical decisions, orchestration failures, verification "
    "outcomes, BKB rule changes, cross-workspace agent behavior patterns, and "
    "architecture lessons. Ignore transient progress notes, raw command output, "
    "drafts, and unverified hypotheses."
)
MANAGER_REFLECT_MISSION = (
    "Act as an evidence-focused supervisor memory for Lethyx. Propose cautious, "
    "auditable behavior rules only when supported by explicit evidence. Prefer "
    "rejecting weak generalizations over broad rules."
)
WORKSPACE_RETAIN_MISSION = (
    "Capture workspace-specific architecture decisions, recurring bugs, "
    "successful fixes, verification reports, accepted BKB rules, and durable "
    "implementation lessons. Ignore low-level transient logs unless they support "
    "a verified lesson."
)
WORKSPACE_REFLECT_MISSION = (
    "Identify recurring workspace patterns and propose narrow BKB candidates "
    "with evidence. Do not propose global manager rules unless evidence spans "
    "multiple runs or workspaces."
)
AGENT_RETAIN_MISSION = (
    "Capture durable behavior lessons about agent systems: tool-use "
    "failures, verification discipline, memory mistakes, and successful patterns. "
    "Ignore task-local implementation details unless they affect agent behavior broadly."
)


@dataclass(frozen=True)
class BankSpec:
    bank_id: str
    name: str
    mission: str


def workspace_bank_id(workspace_id: str) -> str:
    clean = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(workspace_id or "").strip()
    )
    return f"ub:workspace:{clean}" if clean else MANAGER_BANK_ID


def agent_bank_id(agent_name: str) -> str:
    clean = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(agent_name or "generic_agent").strip().lower()
    )
    return f"ub:agent:{clean or 'generic_agent'}"


def bank_specs(*, workspace_id: str = "") -> list[BankSpec]:
    specs = [
        BankSpec(
            bank_id=MANAGER_BANK_ID,
            name="Lethyx Manager Archive",
            mission=MANAGER_RETAIN_MISSION + "\n\nReflect mission: " + MANAGER_REFLECT_MISSION,
        ),
        BankSpec(
            bank_id=GENERIC_AGENT_BANK_ID,
            name="Generic Agent Archive",
            mission=AGENT_RETAIN_MISSION,
        ),
    ]
    if workspace_id:
        specs.append(
            BankSpec(
                bank_id=workspace_bank_id(workspace_id),
                name=f"Lethyx Workspace Archive: {workspace_id}",
                mission=WORKSPACE_RETAIN_MISSION
                + "\n\nReflect mission: "
                + WORKSPACE_REFLECT_MISSION,
            )
        )
    return specs


def ensure_banks(client: Any, *, workspace_id: str = "") -> dict[str, Any]:
    banks = getattr(client, "banks", None)
    if banks is None:
        return {"ok": False, "reason": "client has no banks API"}

    created: list[str] = []
    configured: list[str] = []
    for spec in bank_specs(workspace_id=workspace_id):
        try:
            if hasattr(banks, "create"):
                banks.create(
                    bank_id=spec.bank_id,
                    name=spec.name,
                    mission=spec.mission,
                )
                created.append(spec.bank_id)
        except Exception as exc:
            if "exist" not in str(exc).lower() and "already" not in str(exc).lower():
                raise
        if hasattr(banks, "set_mission"):
            banks.set_mission(bank_id=spec.bank_id, mission=spec.mission)
            configured.append(spec.bank_id)
    return {"ok": True, "created": created, "configured": configured}
