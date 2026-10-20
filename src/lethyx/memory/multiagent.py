"""Parent-owned multi-agent memory scope and durable-service boundary."""

from dataclasses import dataclass, field
from typing import Literal

from lethyx.contracts import EvidenceRef
from lethyx.memory.scopes import ScopeLineage


AgentRole = Literal["parent", "subagent", "verifier"]
MemoryAgentOperation = Literal[
    "read_task",
    "read_subtask",
    "submit_self_report",
    "verify_self_report",
    "durable_write",
    "promotion",
    "project_truth",
    "global_truth",
    "start_backend",
    "start_indexer",
    "start_consolidator",
]

_SUBAGENT_OPERATIONS = frozenset({"read_task", "read_subtask", "submit_self_report"})
_VERIFIER_OPERATIONS = frozenset({"read_task", "read_subtask", "verify_self_report"})
_PARENT_OPERATIONS = frozenset(
    {
        *_SUBAGENT_OPERATIONS,
        "verify_self_report",
        "durable_write",
        "promotion",
        "project_truth",
        "global_truth",
        "start_backend",
        "start_indexer",
        "start_consolidator",
    }
)


def _required(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _evidence(value: object, *, field_name: str) -> tuple[EvidenceRef, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    if not value or any(not isinstance(ref, EvidenceRef) for ref in value):
        raise ValueError(f"{field_name} require EvidenceRef values")
    return tuple(dict.fromkeys(value))


@dataclass(frozen=True)
class AgentExecutionContext:
    agent_id: str
    role: AgentRole
    lineage: ScopeLineage
    parent_agent_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_id",
            _required(self.agent_id, field_name="agent_id"),
        )
        if self.role not in {"parent", "subagent", "verifier"}:
            raise ValueError("agent role is invalid")
        if not isinstance(self.lineage, ScopeLineage):
            raise ValueError("agent context requires ScopeLineage")
        parent = str(self.parent_agent_id or "").strip()
        if self.role == "subagent" and not parent:
            raise ValueError("subagent requires parent_agent_id")
        if self.role == "parent" and parent:
            raise ValueError("parent agent cannot have parent_agent_id")
        object.__setattr__(self, "parent_agent_id", parent)


@dataclass(frozen=True)
class SubagentContextGrant:
    grant_id: str
    parent_agent_id: str
    subagent_id: str
    task_scope_key: str
    subtask_scope_key: str
    allowed_scope_keys: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "grant_id",
            "parent_agent_id",
            "subagent_id",
            "task_scope_key",
            "subtask_scope_key",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        expected = (self.subtask_scope_key, self.task_scope_key)
        if self.allowed_scope_keys != expected:
            raise ValueError("subagent grant may contain only exact subtask and task scopes")
        if not self.subtask_scope_key.startswith("subtask:") or not self.task_scope_key.startswith(
            "task:"
        ):
            raise ValueError("subagent grant scope keys are invalid")
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence(self.evidence_refs, field_name="subagent grant evidence_refs"),
        )


@dataclass(frozen=True)
class SubagentSelfReport:
    report_id: str
    subagent_id: str
    parent_agent_id: str
    task_id: str
    subtask_id: str
    content: str
    evidence_refs: tuple[EvidenceRef, ...]
    status: Literal["candidate"] = field(default="candidate", init=False)
    authority: Literal["advisory"] = field(default="advisory", init=False)
    verification_status: Literal["unverified"] = field(default="unverified", init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "report_id",
            "subagent_id",
            "parent_agent_id",
            "task_id",
            "subtask_id",
            "content",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence(self.evidence_refs, field_name="subagent report evidence_refs"),
        )


@dataclass(frozen=True)
class VerifiedSubagentReport:
    report: SubagentSelfReport
    verifier_agent_id: str
    verifier_role: Literal["parent", "verifier"]
    verification_evidence_refs: tuple[EvidenceRef, ...]
    status: Literal["accepted"] = field(default="accepted", init=False)
    authority: Literal["advisory"] = field(default="advisory", init=False)
    durable_write_authorized: bool = field(default=False, init=False)
    promotion_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.report, SubagentSelfReport):
            raise ValueError("verified report requires SubagentSelfReport")
        object.__setattr__(
            self,
            "verifier_agent_id",
            _required(self.verifier_agent_id, field_name="verifier_agent_id"),
        )
        if self.verifier_role not in {"parent", "verifier"}:
            raise ValueError("self-report verifier must be parent or verifier")
        object.__setattr__(
            self,
            "verification_evidence_refs",
            _evidence(
                self.verification_evidence_refs,
                field_name="self-report verification evidence_refs",
            ),
        )


@dataclass(frozen=True)
class AgentOperationDecision:
    agent_id: str
    operation: str
    allowed: bool
    reason: str
    parent_owned: bool


@dataclass(frozen=True)
class DurableMemoryServiceLease:
    lease_id: str
    owner_agent_id: str
    workspace_id: str
    backend_id: str
    indexer_id: str
    consolidator_id: str
    evidence_refs: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "lease_id",
            "owner_agent_id",
            "workspace_id",
            "backend_id",
            "indexer_id",
            "consolidator_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence(self.evidence_refs, field_name="durable service lease evidence_refs"),
        )


class MultiAgentOwnershipKernel:
    def issue_subagent_grant(
        self,
        subagent: AgentExecutionContext,
        *,
        evidence_refs: tuple[EvidenceRef, ...],
    ) -> SubagentContextGrant:
        if not isinstance(subagent, AgentExecutionContext) or subagent.role != "subagent":
            raise ValueError("subagent grant requires subagent context")
        lineage = subagent.lineage
        if not lineage.task_id or not lineage.subtask_id:
            raise ValueError("subagent grant requires task and subtask lineage")
        return SubagentContextGrant(
            grant_id=f"grant-{subagent.agent_id}-{lineage.subtask_id}",
            parent_agent_id=subagent.parent_agent_id,
            subagent_id=subagent.agent_id,
            task_scope_key=f"task:{lineage.task_id}",
            subtask_scope_key=f"subtask:{lineage.subtask_id}",
            allowed_scope_keys=(
                f"subtask:{lineage.subtask_id}",
                f"task:{lineage.task_id}",
            ),
            evidence_refs=evidence_refs,
        )

    def authorize(
        self,
        actor: AgentExecutionContext,
        operation: MemoryAgentOperation,
    ) -> AgentOperationDecision:
        if not isinstance(actor, AgentExecutionContext):
            raise TypeError("actor must be AgentExecutionContext")
        if operation not in _PARENT_OPERATIONS:
            raise ValueError("memory agent operation is invalid")
        allowed_operations = {
            "parent": _PARENT_OPERATIONS,
            "subagent": _SUBAGENT_OPERATIONS,
            "verifier": _VERIFIER_OPERATIONS,
        }[actor.role]
        allowed = operation in allowed_operations
        parent_owned = operation in _PARENT_OPERATIONS - _SUBAGENT_OPERATIONS - {
            "verify_self_report"
        }
        return AgentOperationDecision(
            agent_id=actor.agent_id,
            operation=operation,
            allowed=allowed,
            reason=("role_operation_allowed" if allowed else "operation_is_parent_owned"),
            parent_owned=parent_owned,
        )

    def verify_report(
        self,
        report: SubagentSelfReport,
        *,
        verifier: AgentExecutionContext,
        evidence_refs: tuple[EvidenceRef, ...],
    ) -> VerifiedSubagentReport:
        decision = self.authorize(verifier, "verify_self_report")
        if not decision.allowed:
            raise ValueError("self-report verification requires parent or verifier")
        if verifier.role == "subagent":
            raise ValueError("self-report verification requires parent or verifier")
        if verifier.role == "parent" and verifier.agent_id != report.parent_agent_id:
            raise ValueError("only the owning parent may verify a subagent report")
        if verifier.role == "verifier" and verifier.parent_agent_id not in {
            "",
            report.parent_agent_id,
        }:
            raise ValueError("verifier is outside the report parent boundary")
        return VerifiedSubagentReport(
            report=report,
            verifier_agent_id=verifier.agent_id,
            verifier_role=verifier.role,
            verification_evidence_refs=evidence_refs,
        )

    def issue_durable_service_lease(
        self,
        owner: AgentExecutionContext,
        *,
        backend_id: str,
        indexer_id: str,
        consolidator_id: str,
        evidence_refs: tuple[EvidenceRef, ...],
    ) -> DurableMemoryServiceLease:
        for operation in ("start_backend", "start_indexer", "start_consolidator"):
            if not self.authorize(owner, operation).allowed:
                raise ValueError("durable memory service is parent-owned")
        return DurableMemoryServiceLease(
            lease_id=f"durable-service-{owner.agent_id}-{owner.lineage.workspace_id}",
            owner_agent_id=owner.agent_id,
            workspace_id=owner.lineage.workspace_id,
            backend_id=backend_id,
            indexer_id=indexer_id,
            consolidator_id=consolidator_id,
            evidence_refs=evidence_refs,
        )
