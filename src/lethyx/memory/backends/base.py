"""Protocol and transport-neutral models for durable memory backends."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol
from lethyx.memory.recall_plan import RetrievalMode

if TYPE_CHECKING:
    from lethyx.memory.backends.contracts import BackendCapabilities
    from lethyx.memory.dependencies import DependencyValidation


class MemoryBusyError(TimeoutError):
    """A storage operation could not acquire its gate within the wait budget."""


class NativeProjectionPending(RuntimeError):
    """Canonical data is durable but the requested native read is not ready."""


@dataclass(frozen=True)
class MemoryQuery:
    query: str
    retrieval_mode: RetrievalMode = "semantic"
    workspace_id: str = ""
    profile_id: str = "default"
    run_id: str = ""
    logical_block_ids: list[str] = field(default_factory=list)
    search_all: bool = False
    record_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_horizon: str = "durable"
    include_candidates: bool = False
    scope_chain: list[str] = field(default_factory=list)
    max_tokens: int = 2048
    limit: int = 10
    temporal_mode: str = "current"
    as_of: str = ""
    known_as_of: str = ""
    revision_id: str = ""
    paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    config_keys: list[str] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)
    artifact_hashes: list[str] = field(default_factory=list)
    facets: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryHit:
    text: str
    source: str
    score: float | None = None
    kind: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconcileReport:
    backend: str
    dry_run: bool
    pending_canonical_heads: int
    pending_write_outbox: int
    repaired: int
    remaining: int
    failures: int
    result_code: str
    reason_code: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "dry_run": self.dry_run,
            "pending_canonical_heads": self.pending_canonical_heads,
            "pending_write_outbox": self.pending_write_outbox,
            "repaired": self.repaired,
            "remaining": self.remaining,
            "failures": self.failures,
            "result_code": self.result_code,
            "reason_code": self.reason_code,
            "timestamp": self.timestamp,
        }


class DurableMemoryBackend(Protocol):
    def health(self) -> dict[str, Any]: ...

    def reconcile(self, *, dry_run: bool = True) -> ReconcileReport: ...

    def capabilities(self) -> "BackendCapabilities": ...

    def ensure_banks(self, *, workspace_id: str = "") -> dict[str, Any]: ...

    def retain_block(self, record: Any) -> dict[str, Any]: ...

    def get_block(self, block_id: str) -> Any | None: ...

    def validate_dependencies(
        self,
        record_id: str,
        *,
        workspace_id: str = "",
        profile_id: str = "default",
        as_of: str = "",
        max_nodes: int = 256,
        max_edges: int = 1024,
    ) -> "DependencyValidation": ...

    def retain_index_node(self, node: Any) -> dict[str, Any]: ...

    def get_index_node(self, node_id: str, *, workspace_id: str = "") -> Any | None: ...

    def select_index_nodes(
        self,
        *,
        workspace_id: str = "",
        node_type: str = "",
        tags: list[str] | tuple[str, ...] | set[str] | None = None,
        limit: int = 50,
    ) -> list[Any]: ...

    def search_index_nodes(
        self,
        *,
        workspace_id: str = "",
        terms: list[str] | tuple[str, ...] = (),
        node_types: list[str] | tuple[str, ...] = (),
        limit: int = 128,
    ) -> list[Any]: ...

    def export_index_nodes(self) -> list[Any]: ...

    def create_prospective_obligation(
        self,
        obligation: Any,
        *,
        event_id: str,
        idempotency_key: str = "",
    ) -> dict[str, Any]: ...

    def get_prospective_obligation(self, obligation_id: str) -> Any | None: ...

    def transition_prospective_obligation(
        self,
        obligation_id: str,
        *,
        expected_version: int,
        expected_status: str,
        status: str,
        evidence_refs: tuple[Any, ...] | list[Any],
        satisfaction_evidence_refs: tuple[Any, ...] | list[Any] = (),
        satisfaction_verifier: str = "",
        satisfaction_verifier_result: str = "",
        event_id: str,
        idempotency_key: str = "",
    ) -> dict[str, Any]: ...

    def prospective_trigger_candidates(
        self,
        context: Any,
        *,
        workspace_id: str = "",
        limit: int = 50,
    ) -> list[Any]: ...

    def recall(self, query: dict[str, Any] | MemoryQuery) -> list[Any]: ...

    def record_rendered_tokens(
        self,
        rendered_tokens_by_record_id: dict[str, int],
    ) -> None: ...

    def expire_scope(self, scope: str, scope_id: str = "", *, run_id: str = "") -> int: ...

    def add_edge(self, edge: Any) -> dict[str, Any]: ...

    def edges_for(self, block_id: str) -> list[Any]: ...

    def export_edges(self) -> list[Any]: ...

    def reverse_provenance_sources(self, derived_id: str) -> tuple[str, ...]: ...

    def rebuild_memory_indexes(self, *, catalog: Any) -> dict[str, Any]: ...

    def refresh_memory_indexes(self, block_id: str) -> dict[str, Any]: ...

    def record_usage(self, usage: Any) -> dict[str, Any]: ...

    def record_memory_decision(self, event: Any) -> dict[str, Any]: ...

    def get_memory_decision(self, decision_id: str) -> Any | None: ...

    def record_memory_usage_event(self, event: Any) -> dict[str, Any]: ...

    def memory_decision_trace(self, decision_id: str) -> Any: ...

    def record_outcome_credit(self, event: Any) -> dict[str, Any]: ...

    def outcome_credits(self, decision_id: str) -> tuple[Any, ...]: ...

    def outcome_credits_for_memory(self, memory_id: str) -> tuple[Any, ...]: ...

    def contextual_utility_summary(self, **kwargs: Any) -> Any: ...

    def value_stats(self, block_id: str) -> dict[str, Any]: ...

    def export_blocks(self) -> list[Any]: ...

    def exact_reference_block_ids(
        self, reference: str, *, workspace_id: str
    ) -> tuple[str, ...]: ...

    def scoped_record_ids(
        self,
        *,
        workspace_id: str,
        profile_id: str,
        scope_key: str,
        logical_block_id: str = "",
    ) -> tuple[str, ...]: ...

    def import_blocks(self, records: list[Any]) -> dict[str, Any]: ...

    def compile_active_overlay(
        self, *, workspace_id: str = "", run_id: str = ""
    ) -> dict[str, Any]: ...

    def render_prompt_items(
        self, *, workspace_id: str = "", run_id: str = ""
    ) -> list[dict[str, Any]]: ...

    def recall_evidence(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryHit]: ...
