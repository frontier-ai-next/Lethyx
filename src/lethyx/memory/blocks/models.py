"""Models for logical memory blocks over stable physical stores."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from lethyx.contracts.memory import (
    DeliveryPolicy,
    MemoryRetention,
    MemoryScope,
    MemoryTemperature,
    _closed_value,
)


_RECALL_PATHS = frozenset(
    {
        "directive_overlay",
        "working_state",
        "hybrid_recall",
        "exact_lookup",
        "reflection_queue",
        "archive_only",
    }
)
_LIFECYCLE_POLICY_FIELDS = frozenset(
    {"initial_status", "mutable", "terminal_statuses", "expires_at"}
)
_ISOLATION_POLICY_FIELDS = frozenset({"dimension", "cross_workspace"})
_ISOLATION_DIMENSIONS = frozenset(
    {"subtask", "task", "session", "workspace", "project", "user_global"}
)


@dataclass(frozen=True)
class MemoryBlock:
    block_id: str
    block_type: str
    temperature: MemoryTemperature
    scope: MemoryScope
    authority: str
    delivery_policy: DeliveryPolicy
    retention: MemoryRetention
    name: str = ""
    tags: tuple[str, ...] = ()
    semantic_memberships: tuple[str, ...] = ()
    guidance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str = ""
    child_ids: tuple[str, ...] = ()
    index_only: bool = False
    agent_writable: bool = True
    owner: str = "Lethyx"
    producer: str = ""
    consumers: tuple[str, ...] = ()
    recall_path: str = ""
    lifecycle_policy: dict[str, Any] = field(default_factory=dict)
    isolation_policy: dict[str, Any] = field(default_factory=dict)
    applies_to: dict[str, Any] = field(default_factory=dict)
    retrieval_policy: dict[str, Any] = field(default_factory=dict)
    graph_policy: dict[str, Any] = field(default_factory=dict)
    promotion_policy: dict[str, Any] = field(default_factory=dict)
    token_budget: int = 1000
    status_policy: dict[str, Any] = field(default_factory=dict)

    @property
    def related_block_ids(self) -> tuple[str, ...]:
        """Return explicit catalog relations without inferring semantic affinity."""

        value = self.graph_policy.get("related_block_ids", ())
        candidates: Iterable[object]
        if isinstance(value, str):
            candidates = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple, set, frozenset)):
            candidates = value
        elif value is None:
            candidates = ()
        else:
            candidates = (value,)
        return tuple(dict.fromkeys(str(item).strip() for item in candidates if str(item).strip()))

    def __post_init__(self) -> None:
        for field_name, shape in (
            ("temperature", MemoryTemperature),
            ("scope", MemoryScope),
            ("delivery_policy", DeliveryPolicy),
            ("retention", MemoryRetention),
        ):
            object.__setattr__(
                self,
                field_name,
                _closed_value(getattr(self, field_name), field_name=field_name, shape=shape),
            )
        object.__setattr__(self, "block_id", str(self.block_id or "").strip())
        object.__setattr__(self, "name", str(self.name or self.block_id).strip())
        object.__setattr__(self, "parent_id", str(self.parent_id or "").strip())
        if not self.name:
            raise ValueError("memory block name is required")
        object.__setattr__(
            self,
            "tags",
            tuple(dict.fromkeys(str(item).strip() for item in self.tags if str(item).strip())),
        )
        object.__setattr__(
            self,
            "child_ids",
            tuple(dict.fromkeys(str(item).strip() for item in self.child_ids if str(item).strip())),
        )
        object.__setattr__(
            self,
            "semantic_memberships",
            tuple(
                dict.fromkeys(
                    str(item).strip() for item in self.semantic_memberships if str(item).strip()
                )
            ),
        )
        object.__setattr__(self, "producer", str(self.producer or "").strip())
        object.__setattr__(
            self,
            "consumers",
            tuple(dict.fromkeys(str(item).strip() for item in self.consumers if str(item).strip())),
        )
        object.__setattr__(self, "recall_path", str(self.recall_path or "").strip())
        if self.recall_path and self.recall_path not in _RECALL_PATHS:
            raise ValueError("unsupported memory block recall_path")
        object.__setattr__(self, "index_only", bool(self.index_only))
        object.__setattr__(self, "agent_writable", bool(self.agent_writable))
        object.__setattr__(self, "token_budget", max(0, int(self.token_budget or 0)))
        if not isinstance(self.applies_to, dict):
            raise ValueError("applies_to must be a dictionary")
        unknown_applies_to = sorted(set(self.applies_to) - {"agents", "workspaces"})
        if unknown_applies_to:
            raise ValueError("unknown applies_to fields: " + ", ".join(unknown_applies_to))
        object.__setattr__(self, "applies_to", dict(self.applies_to))
        lifecycle_policy = dict(self.lifecycle_policy or {})
        unknown_lifecycle = sorted(set(lifecycle_policy) - _LIFECYCLE_POLICY_FIELDS)
        if unknown_lifecycle:
            raise ValueError("unknown lifecycle_policy fields: " + ", ".join(unknown_lifecycle))
        if lifecycle_policy:
            initial_status = str(lifecycle_policy.get("initial_status") or "").strip()
            if not initial_status:
                raise ValueError("lifecycle_policy initial_status is required")
            if type(lifecycle_policy.get("mutable")) is not bool:
                raise ValueError("lifecycle_policy mutable must be a bool")
            terminal_statuses = lifecycle_policy.get("terminal_statuses") or ()
            if isinstance(terminal_statuses, str):
                terminal_statuses = (terminal_statuses,)
            elif not isinstance(terminal_statuses, (list, tuple, set, frozenset)):
                raise ValueError("lifecycle_policy terminal_statuses must be a sequence")
            normalized_terminal_statuses = tuple(
                dict.fromkeys(str(item).strip() for item in terminal_statuses if str(item).strip())
            )
            if not normalized_terminal_statuses:
                raise ValueError("lifecycle_policy terminal_statuses are required")
            lifecycle_policy["initial_status"] = initial_status
            lifecycle_policy["terminal_statuses"] = normalized_terminal_statuses
            expires_at = str(lifecycle_policy.get("expires_at") or "").strip()
            if expires_at:
                try:
                    parsed_expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError("lifecycle_policy expires_at must be ISO-8601") from exc
                if parsed_expiry.tzinfo is None:
                    parsed_expiry = parsed_expiry.replace(tzinfo=timezone.utc)
                lifecycle_policy["expires_at"] = parsed_expiry.astimezone(timezone.utc).isoformat()
            else:
                lifecycle_policy.pop("expires_at", None)
        object.__setattr__(self, "lifecycle_policy", lifecycle_policy)
        isolation_policy = dict(self.isolation_policy or {})
        unknown_isolation = sorted(set(isolation_policy) - _ISOLATION_POLICY_FIELDS)
        if unknown_isolation:
            raise ValueError("unknown isolation_policy fields: " + ", ".join(unknown_isolation))
        dimension = str(isolation_policy.get("dimension") or "").strip()
        if isolation_policy:
            if not dimension:
                raise ValueError("isolation_policy dimension is required")
            if dimension not in _ISOLATION_DIMENSIONS:
                raise ValueError("unsupported memory block isolation dimension")
            if type(isolation_policy.get("cross_workspace")) is not bool:
                raise ValueError("isolation_policy cross_workspace must be a bool")
            isolation_policy["dimension"] = dimension
        object.__setattr__(self, "isolation_policy", isolation_policy)
        for field_name in (
            "guidance",
            "metadata",
            "retrieval_policy",
            "graph_policy",
            "promotion_policy",
            "status_policy",
        ):
            object.__setattr__(self, field_name, dict(getattr(self, field_name) or {}))
        unknown_guidance = sorted(set(self.guidance) - {"synopsis", "use_when", "avoid_when"})
        if unknown_guidance:
            raise ValueError("unknown guidance fields: " + ", ".join(unknown_guidance))
