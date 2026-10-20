import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast


ActiveMemoryMaskStatus = Literal["active", "suspended", "expired"]
ActiveMemorySelectorType = Literal[
    "pinned_records",
    "canonical_heads",
    "latest_records",
    "backend_query",
    "working_state",
]

_MASK_STATUSES = frozenset({"active", "suspended", "expired"})
_SELECTOR_TYPES = frozenset(
    {
        "pinned_records",
        "canonical_heads",
        "latest_records",
        "backend_query",
        "working_state",
    }
)


def _required_text(value: object, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _text_tuple(values: object, *, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")
    return tuple(dict.fromkeys(_required_text(value, field_name=field_name) for value in values))


def _optional_time(value: object, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return normalized


@dataclass(frozen=True)
class ActiveMemorySelector:
    block_id: str
    selector_type: ActiveMemorySelectorType
    record_ids: tuple[str, ...] = ()
    query: str = ""
    limit: int = 8
    working_state_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _required_text(self.block_id, field_name="block_id"))
        if self.selector_type not in _SELECTOR_TYPES:
            raise ValueError("selector_type is unsupported")
        object.__setattr__(
            self,
            "record_ids",
            _text_tuple(self.record_ids, field_name="record_ids"),
        )
        object.__setattr__(self, "query", str(self.query or "").strip())
        object.__setattr__(self, "working_state_id", str(self.working_state_id or "").strip())
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError("selector limit must be a non-negative int")
        if self.selector_type == "pinned_records" and not self.record_ids:
            raise ValueError("pinned_records selector requires record_ids")
        if self.selector_type == "backend_query" and not self.query:
            raise ValueError("backend_query selector requires query")
        if self.selector_type != "backend_query" and self.query:
            raise ValueError("query is only valid for backend_query selectors")
        if self.selector_type != "working_state" and self.working_state_id:
            raise ValueError("working_state_id is only valid for working_state selectors")

    @classmethod
    def from_mapping(cls, value: object) -> "ActiveMemorySelector":
        if not isinstance(value, dict):
            raise ValueError("active memory selector must be a mapping")
        selector_type = str(value.get("selector_type") or "")
        if selector_type not in _SELECTOR_TYPES:
            raise ValueError("selector_type is unsupported")
        return cls(
            block_id=str(value.get("block_id") or ""),
            selector_type=cast(ActiveMemorySelectorType, selector_type),
            record_ids=tuple(value.get("record_ids") or ()),
            query=str(value.get("query") or ""),
            limit=int(value.get("limit", 8)),
            working_state_id=str(value.get("working_state_id") or ""),
        )


@dataclass(frozen=True)
class ActiveMemoryPriority:
    block_id: str
    priority: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _required_text(self.block_id, field_name="block_id"))
        if type(self.priority) is not int:
            raise ValueError("priority must be an int")

    @classmethod
    def from_mapping(cls, value: object) -> "ActiveMemoryPriority":
        if not isinstance(value, dict):
            raise ValueError("active memory priority must be a mapping")
        return cls(
            block_id=str(value.get("block_id") or ""),
            priority=int(value.get("priority", 0)),
        )


@dataclass(frozen=True)
class ActiveMemoryBlockBudget:
    block_id: str
    token_budget: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _required_text(self.block_id, field_name="block_id"))
        if type(self.token_budget) is not int or self.token_budget < 0:
            raise ValueError("block token_budget must be a non-negative int")

    @classmethod
    def from_mapping(cls, value: object) -> "ActiveMemoryBlockBudget":
        if not isinstance(value, dict):
            raise ValueError("active memory block budget must be a mapping")
        return cls(
            block_id=str(value.get("block_id") or ""),
            token_budget=int(value.get("token_budget", 0)),
        )


@dataclass(frozen=True)
class ActiveMemoryMask:
    workspace_id: str
    profile_id: str
    revision: int
    status: ActiveMemoryMaskStatus
    selected_block_ids: tuple[str, ...]
    token_budget: int
    activation_reason: str
    session_id: str = ""
    task_id: str = ""
    context_refs: tuple[str, ...] = ()
    pinned_record_ids: tuple[str, ...] = ()
    selectors: tuple[ActiveMemorySelector, ...] = ()
    priorities: tuple[ActiveMemoryPriority, ...] = ()
    block_token_budgets: tuple[ActiveMemoryBlockBudget, ...] = ()
    lifecycle: str = "durable"
    created_at: str = ""
    updated_at: str = ""
    expires_at: str = ""
    catalog_revision: int = 0
    mask_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workspace_id",
            _required_text(self.workspace_id, field_name="workspace_id"),
        )
        object.__setattr__(
            self,
            "profile_id",
            _required_text(self.profile_id, field_name="profile_id"),
        )
        object.__setattr__(self, "session_id", str(self.session_id or "").strip())
        object.__setattr__(self, "task_id", str(self.task_id or "").strip())
        object.__setattr__(
            self,
            "activation_reason",
            _required_text(
                self.activation_reason,
                field_name="activation_reason",
            ),
        )
        object.__setattr__(
            self, "lifecycle", _required_text(self.lifecycle, field_name="lifecycle")
        )
        object.__setattr__(self, "mask_id", _required_text(self.mask_id, field_name="mask_id"))
        try:
            uuid.UUID(self.mask_id)
        except ValueError as exc:
            raise ValueError("mask_id must be a UUID") from exc
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("mask revision must be a positive int")
        if self.status not in _MASK_STATUSES:
            raise ValueError("active memory mask status is unsupported")
        if type(self.token_budget) is not int or self.token_budget < 0:
            raise ValueError("mask token_budget must be a non-negative int")
        if type(self.catalog_revision) is not int or self.catalog_revision < 0:
            raise ValueError("catalog_revision must be a non-negative int")
        for field_name in (
            "selected_block_ids",
            "context_refs",
            "pinned_record_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _text_tuple(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "selectors", tuple(self.selectors or ()))
        object.__setattr__(self, "priorities", tuple(self.priorities or ()))
        object.__setattr__(self, "block_token_budgets", tuple(self.block_token_budgets or ()))
        if any(not isinstance(value, ActiveMemorySelector) for value in self.selectors):
            raise TypeError("selectors must contain ActiveMemorySelector values")
        if any(not isinstance(value, ActiveMemoryPriority) for value in self.priorities):
            raise TypeError("priorities must contain ActiveMemoryPriority values")
        if any(
            not isinstance(value, ActiveMemoryBlockBudget) for value in self.block_token_budgets
        ):
            raise TypeError("block_token_budgets must contain ActiveMemoryBlockBudget values")
        selected = set(self.selected_block_ids)
        referenced_blocks = {
            *(selector.block_id for selector in self.selectors),
            *(priority.block_id for priority in self.priorities),
            *(budget.block_id for budget in self.block_token_budgets),
        }
        if referenced_blocks.difference(selected):
            raise ValueError(
                "mask selectors, priorities, and budgets must reference selected blocks"
            )
        object.__setattr__(
            self,
            "created_at",
            _optional_time(self.created_at, field_name="created_at"),
        )
        object.__setattr__(
            self,
            "updated_at",
            _optional_time(self.updated_at, field_name="updated_at"),
        )
        object.__setattr__(
            self,
            "expires_at",
            _optional_time(self.expires_at, field_name="expires_at"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def is_expired(self, *, at: datetime | None = None) -> bool:
        """Return whether lifecycle state or the explicit deadline ended this mask."""

        if self.status == "expired":
            return True
        if not self.expires_at:
            return False
        reference = at or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            raise ValueError("active memory expiry comparison requires a timezone")
        expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        return expiry <= reference

    @classmethod
    def from_mapping(cls, value: object) -> "ActiveMemoryMask":
        if not isinstance(value, dict):
            raise ValueError("active memory mask must be a mapping")
        status = str(value.get("status") or "")
        if status not in _MASK_STATUSES:
            raise ValueError("active memory mask status is unsupported")
        return cls(
            mask_id=str(value.get("mask_id") or ""),
            workspace_id=str(value.get("workspace_id") or ""),
            profile_id=str(value.get("profile_id") or ""),
            session_id=str(value.get("session_id") or ""),
            task_id=str(value.get("task_id") or ""),
            context_refs=tuple(value.get("context_refs") or ()),
            revision=int(value.get("revision", 0)),
            status=cast(ActiveMemoryMaskStatus, status),
            selected_block_ids=tuple(value.get("selected_block_ids") or ()),
            pinned_record_ids=tuple(value.get("pinned_record_ids") or ()),
            selectors=tuple(
                ActiveMemorySelector.from_mapping(item) for item in value.get("selectors") or ()
            ),
            priorities=tuple(
                ActiveMemoryPriority.from_mapping(item) for item in value.get("priorities") or ()
            ),
            token_budget=int(value.get("token_budget", 0)),
            block_token_budgets=tuple(
                ActiveMemoryBlockBudget.from_mapping(item)
                for item in value.get("block_token_budgets") or ()
            ),
            lifecycle=str(value.get("lifecycle") or "durable"),
            activation_reason=str(value.get("activation_reason") or ""),
            created_at=str(value.get("created_at") or ""),
            updated_at=str(value.get("updated_at") or ""),
            expires_at=str(value.get("expires_at") or ""),
            catalog_revision=int(value.get("catalog_revision", 0)),
        )
