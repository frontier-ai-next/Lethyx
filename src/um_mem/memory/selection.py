"""Policy for selecting memory blocks by canonical memory metadata."""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from um_mem.contracts import MemoryBlockRecord, is_protected_memory
from um_mem.contracts.memory import has_decisive_evidence, memory_scope_key
from um_mem.contracts.models import APPROVED_TRUST_LEVELS
from um_mem.memory.recall_plan import SourceHorizon
from um_mem.memory.scopes import OrderedScopeChain

_EXCLUDED_STATUSES = {"quarantined", "superseded", "deprecated", "retracted"}
_SOURCE_HORIZONS = frozenset({"working", "durable", "archive"})


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_time(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class SelectionContext:
    """Inputs used to decide whether a memory block may be used now."""

    query: str = ""
    scope_chain: tuple[str, ...] = ()
    run_id: str = ""
    source_horizon: SourceHorizon = "durable"
    include_candidates: bool = False
    historical: bool = False
    as_of: str = ""
    delivery_policy: str = "on_demand"
    max_blocks: int = 10
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.as_of, str) or (self.as_of and _parse_time(self.as_of) is None):
            raise ValueError("SelectionContext as_of must be an ISO timestamp or empty")
        if self.scope_chain:
            object.__setattr__(
                self,
                "scope_chain",
                OrderedScopeChain.from_keys(tuple(self.scope_chain)).keys,
            )
        if self.source_horizon not in _SOURCE_HORIZONS:
            raise ValueError("source_horizon must be working, durable, or archive")
        if not isinstance(self.tags, tuple):
            object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))


class MemorySelectionPolicy:
    """Select canonical advisory or directive memory from block metadata.

    The policy is deliberately backend-neutral: it does not call storage systems,
    and it treats the selected storage backend as an implementation detail. A
    block becomes usable only when its status, delivery, temperature, scope, evidence,
    and workspace match the requested context.
    """

    def eligible_for_recall(self, block: MemoryBlockRecord, ctx: SelectionContext) -> bool:
        if not self._scope_matches(block, ctx):
            return False
        if is_protected_memory(block.sensitivity, block.content_mode, block.operational_risk):
            return False
        if self._index_only(block):
            return False
        if block.status in _EXCLUDED_STATUSES and not ctx.historical:
            return False
        if block.status == "candidate" and not ctx.include_candidates:
            return False
        if (
            block.status == "candidate"
            and block.run_id != ctx.run_id
            and not self._cross_run_agent_memory(block)
        ):
            return False
        if not self._validity_matches(block, ctx):
            return False
        return self._within_source_horizon(block, ctx.source_horizon)

    def eligible_for_overlay(self, block: MemoryBlockRecord, ctx: SelectionContext) -> bool:
        if not self.eligible_for_recall(block, ctx):
            return False
        if block.delivery_policy == "always_on":
            return True
        return block.delivery_policy in {"working_set", "on_demand"}

    def eligible_for_directive(self, block: MemoryBlockRecord, ctx: SelectionContext) -> bool:
        if ctx.historical:
            return False
        if not self.eligible_for_overlay(block, ctx):
            return False
        if block.authority != "directive":
            return False
        if block.status != "accepted":
            return False
        if block.verification_status != "verified" or not block.verification_methods:
            return False
        if block.trust_level not in APPROVED_TRUST_LEVELS:
            return False
        if not has_decisive_evidence(block.evidence_refs):
            return False
        return block.promotion_decision == "accepted"

    def desired_activity(self, block: MemoryBlockRecord) -> str:
        if block.retention == "archive":
            return "archive"
        if block.delivery_policy in {"reflection_only", "hidden"} or block.temperature == "cold":
            return "passive"
        return "active"

    def _scope_matches(self, block: MemoryBlockRecord, ctx: SelectionContext) -> bool:
        if not ctx.scope_chain:
            return False
        try:
            scope_key = memory_scope_key(block)
        except ValueError:
            return False
        return scope_key in ctx.scope_chain

    def _within_source_horizon(
        self,
        block: MemoryBlockRecord,
        source_horizon: SourceHorizon,
    ) -> bool:
        if block.delivery_policy == "hidden":
            return False
        if source_horizon == "archive":
            return True
        if source_horizon == "durable":
            return block.retention != "archive"
        return block.retention in {"ephemeral", "session"} or block.delivery_policy in {
            "always_on",
            "working_set",
        }

    def _index_only(self, block: MemoryBlockRecord) -> bool:
        metadata = dict(block.metadata or {})
        return block.kind == "skill_index" or _truthy(
            metadata.get("index_only") or metadata.get("is_index_only")
        )

    def _cross_run_agent_memory(self, block: MemoryBlockRecord) -> bool:
        """Allow an explicit agent remember as durable supplemental evidence."""

        metadata = dict(block.metadata or {})
        return bool(
            str(metadata.get("evidence_kind") or "").strip()
            in {"agent_tool_claim", "user_statement", "user_stated"}
            and block.retention == "durable"
            and block.scope in {"workspace", "project", "user_global"}
            and block.authority == "advisory"
            and block.trust_level == "agent_reported"
            and block.verification_status == "unverified"
            and block.promotion_decision == "not_requested"
            and block.delivery_policy in {"working_set", "on_demand"}
        )

    def _validity_matches(self, block: MemoryBlockRecord, ctx: SelectionContext) -> bool:
        if ctx.historical and not ctx.as_of:
            return True
        bound = _parse_time(ctx.as_of) if ctx.as_of else datetime.now(UTC)
        if bound is None:
            raise ValueError("SelectionContext as_of must be an ISO timestamp or empty")
        expires_at = _parse_time(block.expires_at)
        if expires_at is not None and expires_at <= bound:
            return False
        valid_from = _parse_time(block.valid_from)
        valid_to = _parse_time(block.valid_to)
        if valid_from is not None and valid_from > bound:
            return False
        return valid_to is None or bound < valid_to
