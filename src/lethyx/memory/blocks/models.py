"""Models for logical memory blocks over stable physical stores."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryBlock:
    block_id: str
    block_type: str
    store: str
    tier: str
    scope: str
    authority: str
    default_surface: str
    tags: tuple[str, ...] = ()
    owner: str = "Lethyx"
    applies_to: dict[str, Any] = field(default_factory=dict)
    retrieval_policy: dict[str, Any] = field(default_factory=dict)
    graph_policy: dict[str, Any] = field(default_factory=dict)
    promotion_policy: dict[str, Any] = field(default_factory=dict)
    token_budget: int = 1000
    lifecycle_policy: dict[str, Any] = field(default_factory=dict)
