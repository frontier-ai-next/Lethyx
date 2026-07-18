"""Null agent policy — a simple stub for testing and graceful degradation."""

from typing import Any


class NullPolicy:
    """Policy that always emits a fixed 'done' action.

    Used when no concrete AgentPolicy is available (e.g. dataset unavailable,
    environment stub mode).
    """

    def act(self, obs: Any, store: Any) -> str:
        return "done"
