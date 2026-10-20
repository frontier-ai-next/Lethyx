"""Lethyx: framework-neutral memory for agent systems."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lethyx.api import (
        AgentSession,
        MemoryClient,
        RememberResult,
        recall,
        remember,
    )
    from lethyx.memory.activation_lease import ActivationLease
    from lethyx.memory.active_mask import ActiveMemoryMask, ActiveMemorySelector
    from lethyx.memory.backends.base import MemoryBusyError
    from lethyx.memory.backends.contracts import BackendCapabilities
    from lethyx.memory.deep_reader import DeepReadResult
    from lethyx.memory.dependencies import DependencyIssue, DependencyValidation
    from lethyx.memory.mutation import MemoryRetractionReceipt
    from lethyx.memory.recall_explanation import RecallExplanation
    from lethyx.memory.record_refs import PinnedRecordRef
    from lethyx.memory.working_set import (
        WorkingRecordRef,
        WorkingRecordSnapshot,
        WorkingSetRead,
        WorkingSnapshotBudgetError,
    )

__version__ = "0.5.0"

_EXPORTS = {
    "AgentSession": ("lethyx.api", "AgentSession"),
    "ActiveMemoryMask": ("lethyx.memory.active_mask", "ActiveMemoryMask"),
    "ActiveMemorySelector": ("lethyx.memory.active_mask", "ActiveMemorySelector"),
    "ActivationLease": ("lethyx.memory.activation_lease", "ActivationLease"),
    "DeepReadResult": ("lethyx.memory.deep_reader", "DeepReadResult"),
    "DependencyIssue": ("lethyx.memory.dependencies", "DependencyIssue"),
    "DependencyValidation": ("lethyx.memory.dependencies", "DependencyValidation"),
    "MemoryClient": ("lethyx.api", "MemoryClient"),
    "MemoryBusyError": ("lethyx.memory.backends.base", "MemoryBusyError"),
    "BackendCapabilities": ("lethyx.memory.backends.contracts", "BackendCapabilities"),
    "PinnedRecordRef": ("lethyx.memory.record_refs", "PinnedRecordRef"),
    "MemoryRetractionReceipt": ("lethyx.memory.mutation", "MemoryRetractionReceipt"),
    "RecallExplanation": ("lethyx.memory.recall_explanation", "RecallExplanation"),
    "RememberResult": ("lethyx.api", "RememberResult"),
    "WorkingRecordRef": ("lethyx.memory.working_set", "WorkingRecordRef"),
    "WorkingRecordSnapshot": ("lethyx.memory.working_set", "WorkingRecordSnapshot"),
    "WorkingSetRead": ("lethyx.memory.working_set", "WorkingSetRead"),
    "WorkingSnapshotBudgetError": ("lethyx.memory.working_set", "WorkingSnapshotBudgetError"),
    "recall": ("lethyx.api", "recall"),
    "remember": ("lethyx.api", "remember"),
}


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "AgentSession",
    "ActiveMemoryMask",
    "ActiveMemorySelector",
    "ActivationLease",
    "DeepReadResult",
    "DependencyIssue",
    "DependencyValidation",
    "MemoryClient",
    "MemoryBusyError",
    "BackendCapabilities",
    "PinnedRecordRef",
    "MemoryRetractionReceipt",
    "RecallExplanation",
    "RememberResult",
    "WorkingRecordRef",
    "WorkingRecordSnapshot",
    "WorkingSetRead",
    "WorkingSnapshotBudgetError",
    "recall",
    "remember",
    "__version__",
]
