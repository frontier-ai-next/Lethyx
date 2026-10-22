"""Typed scope lineage, ordered lookup chains, and evidence precedence."""

from dataclasses import dataclass
from typing import Literal, cast

ScopeKind = Literal["subtask", "task", "session", "workspace", "project", "user_global"]
MemoryPrecedenceClass = Literal[
    "verified_workspace_fact",
    "workspace_decision",
    "workspace_procedure",
    "explicit_user_global_preference",
    "inferred_global_candidate",
]

_SCOPE_RANK: dict[str, int] = {
    "subtask": 0,
    "task": 1,
    "session": 2,
    "workspace": 3,
    "project": 4,
    "user_global": 5,
}
_PRECEDENCE_RANK: dict[str, int] = {
    "verified_workspace_fact": 0,
    "workspace_decision": 1,
    "workspace_procedure": 2,
    "explicit_user_global_preference": 3,
    "inferred_global_candidate": 4,
}


def _required(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional(value: object, *, field_name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value.strip()


@dataclass(frozen=True, kw_only=True)
class ScopeLineage:
    workspace_id: str
    project_id: str = ""
    session_id: str = ""
    task_id: str = ""
    subtask_id: str = ""
    user_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workspace_id",
            _required(self.workspace_id, field_name="workspace_id"),
        )
        for field_name in (
            "project_id",
            "session_id",
            "task_id",
            "subtask_id",
            "user_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional(getattr(self, field_name), field_name=field_name),
            )
        for child, parent in (
            ("task_id", "session_id"),
            ("subtask_id", "task_id"),
        ):
            if getattr(self, child) and not getattr(self, parent):
                raise ValueError(f"{child} requires {parent}")


@dataclass(frozen=True)
class ScopeAddress:
    kind: ScopeKind
    scope_id: str

    def __post_init__(self) -> None:
        if self.kind not in _SCOPE_RANK:
            raise ValueError("scope address kind is invalid")
        scope_id = _required(self.scope_id, field_name="scope_id")
        if scope_id == "*":
            raise ValueError("scope address cannot use a wildcard identity")
        object.__setattr__(
            self,
            "scope_id",
            scope_id,
        )

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.scope_id}"


@dataclass(frozen=True)
class OrderedScopeChain:
    entries: tuple[ScopeAddress, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, ScopeAddress) for entry in self.entries
        ):
            raise ValueError("scope chain entries must be ScopeAddress values")
        if not self.entries:
            raise ValueError("scope chain cannot be empty")
        keys = tuple(entry.key for entry in self.entries)
        if len(set(keys)) != len(keys):
            raise ValueError("scope chain cannot contain duplicates")
        kinds = tuple(entry.kind for entry in self.entries)
        if len(set(kinds)) != len(kinds):
            raise ValueError("scope chain cannot contain multiple identities for one scope kind")
        ranks = tuple(_SCOPE_RANK[entry.kind] for entry in self.entries)
        if ranks != tuple(sorted(ranks)):
            raise ValueError("scope chain must be ordered from narrow to broad")

    @classmethod
    def from_lineage(
        cls,
        lineage: ScopeLineage,
        *,
        include_user_global: bool = False,
    ) -> "OrderedScopeChain":
        if not isinstance(lineage, ScopeLineage):
            raise TypeError("lineage must be ScopeLineage")
        if not isinstance(include_user_global, bool):
            raise ValueError("include_user_global must be a bool")
        entries: list[ScopeAddress] = []
        scope_fields: tuple[tuple[ScopeKind, str], ...] = (
            ("subtask", "subtask_id"),
            ("task", "task_id"),
            ("session", "session_id"),
            ("workspace", "workspace_id"),
            ("project", "project_id"),
        )
        for kind, field_name in scope_fields:
            scope_id = getattr(lineage, field_name)
            if scope_id:
                entries.append(ScopeAddress(kind=kind, scope_id=scope_id))
        if include_user_global:
            if not lineage.user_id:
                raise ValueError("user-global scope requires user_id")
            entries.append(ScopeAddress(kind="user_global", scope_id=lineage.user_id))
        return cls(entries=tuple(entries))

    @classmethod
    def from_keys(cls, keys: tuple[str, ...]) -> "OrderedScopeChain":
        if not isinstance(keys, tuple):
            raise ValueError("scope chain keys must be a tuple")
        entries: list[ScopeAddress] = []
        for key in keys:
            if not isinstance(key, str) or ":" not in key:
                raise ValueError("scope chain key must use kind:scope_id")
            kind, scope_id = key.split(":", 1)
            if kind not in _SCOPE_RANK:
                raise ValueError("scope chain key has an invalid kind")
            entries.append(ScopeAddress(kind=cast(ScopeKind, kind), scope_id=scope_id))
        return cls(entries=tuple(entries))

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(entry.key for entry in self.entries)

    def includes(self, kind: ScopeKind) -> bool:
        return any(entry.kind == kind for entry in self.entries)


def bind_scope_chain(
    lineage: ScopeLineage,
    requested: tuple[str, ...],
) -> tuple[str, ...]:
    """Bind an exact requested chain to trusted host-provided lineage."""

    if not isinstance(lineage, ScopeLineage):
        raise TypeError("lineage must be ScopeLineage")
    if not isinstance(requested, tuple):
        raise ValueError("requested scope chain must be a tuple")
    local = OrderedScopeChain.from_lineage(lineage).keys
    if not requested:
        return local
    selected = OrderedScopeChain.from_keys(requested).keys
    allowed = set(local)
    if lineage.user_id:
        allowed.add(f"user_global:{lineage.user_id}")
    if any(scope_key not in allowed for scope_key in selected):
        raise ValueError("scope chain is not bound to caller lineage")
    return selected


@dataclass(frozen=True)
class MemoryPrecedenceCandidate:
    candidate_id: str
    precedence_class: MemoryPrecedenceClass
    scope: Literal["workspace", "user_global"]
    verified: bool
    explicit: bool
    score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _required(self.candidate_id, field_name="candidate_id"),
        )
        if self.precedence_class not in _PRECEDENCE_RANK:
            raise ValueError("memory precedence class is invalid")
        if self.scope not in {"workspace", "user_global"}:
            raise ValueError("precedence candidate scope is invalid")
        if not isinstance(self.verified, bool) or not isinstance(self.explicit, bool):
            raise ValueError("precedence evidence flags must be bools")
        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise ValueError("precedence score must be numeric")
        object.__setattr__(self, "score", float(self.score))
        if self.precedence_class.startswith("workspace_") and self.scope != "workspace":
            raise ValueError("workspace precedence requires workspace scope")
        if self.precedence_class == "verified_workspace_fact" and (
            self.scope != "workspace" or not self.verified
        ):
            raise ValueError("verified workspace fact requires verified workspace evidence")
        if self.precedence_class == "explicit_user_global_preference" and (
            self.scope != "user_global" or not self.explicit
        ):
            raise ValueError("explicit global preference requires explicit user-global evidence")
        if self.precedence_class == "inferred_global_candidate" and (
            self.scope != "user_global" or self.explicit
        ):
            raise ValueError("inferred global candidate must remain non-explicit user-global")


def order_by_memory_precedence(
    candidates: tuple[MemoryPrecedenceCandidate, ...],
) -> tuple[MemoryPrecedenceCandidate, ...]:
    if not isinstance(candidates, tuple) or any(
        not isinstance(candidate, MemoryPrecedenceCandidate) for candidate in candidates
    ):
        raise ValueError("candidates must contain MemoryPrecedenceCandidate values")
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                _PRECEDENCE_RANK[candidate.precedence_class],
                -candidate.score,
                candidate.candidate_id,
            ),
        )
    )
