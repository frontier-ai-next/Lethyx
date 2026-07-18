from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class RecallBundle:
    always_on: list[dict[str, Any]] = field(default_factory=list)
    hot: list[dict[str, Any]] = field(default_factory=list)
    warm: list[dict[str, Any]] = field(default_factory=list)
    graph_neighbours: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "always_on": self.always_on,
            "hot": self.hot,
            "warm": self.warm,
            "graph_neighbours": self.graph_neighbours,
        }

    def all_nodes(self) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for group in (self.always_on, self.hot, self.warm, self.graph_neighbours):
            for node in group:
                nid = node.get("id", "")
                if nid not in seen:
                    seen.add(nid)
                    result.append(node)
        return result
