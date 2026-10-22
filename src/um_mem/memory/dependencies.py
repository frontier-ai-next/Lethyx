"""Read-only validity receipts for exact canonical record dependencies."""

from dataclasses import dataclass
from typing import Literal

DependencyIssueCode = Literal[
    "record_unavailable",
    "profile_mismatch",
    "lifecycle_invalid",
    "revision_not_current",
    "outside_validity",
    "dependency_cycle",
    "dependency_budget_exhausted",
]


@dataclass(frozen=True)
class DependencyIssue:
    record_id: str
    code: DependencyIssueCode


@dataclass(frozen=True)
class DependencyValidation:
    """Canonical DERIVED_FROM validity, without a semantic evidence judgment.

    Index nodes and external evidence are not canonical record dependencies.
    An incomplete traversal does not establish that a record is current.
    """

    record_id: str
    workspace_id: str
    profile_id: str
    as_of: str
    status: Literal["valid", "invalidated", "incomplete"]
    checked_record_ids: tuple[str, ...]
    issues: tuple[DependencyIssue, ...]
    max_nodes: int
    max_edges: int
