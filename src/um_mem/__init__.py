"""UM-MEM: framework-neutral memory for agent systems."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from um_mem.api import (
        AgentSession,
        MemoryClient,
        RememberResult,
        recall,
        remember,
    )
    from um_mem.memory.activation_lease import ActivationLease
    from um_mem.memory.active_mask import ActiveMemoryMask, ActiveMemorySelector
    from um_mem.memory.backends.base import MemoryBusyError
    from um_mem.memory.backends.contracts import BackendCapabilities
    from um_mem.memory.deep_reader import DeepReadResult
    from um_mem.memory.dependencies import DependencyIssue, DependencyValidation
    from um_mem.memory.mutation import MemoryRetractionReceipt
    from um_mem.memory.recall_explanation import RecallExplanation
    from um_mem.memory.record_refs import PinnedRecordRef
    from um_mem.memory.working_set import (
        WorkingRecordRef,
        WorkingRecordSnapshot,
        WorkingSetRead,
        WorkingSnapshotBudgetError,
    )

__version__ = "0.6.0"

_EXPORTS = {
    "AgentSession": ("um_mem.api", "AgentSession"),
    "ActiveMemoryMask": ("um_mem.memory.active_mask", "ActiveMemoryMask"),
    "ActiveMemorySelector": ("um_mem.memory.active_mask", "ActiveMemorySelector"),
    "ActivationLease": ("um_mem.memory.activation_lease", "ActivationLease"),
    "DeepReadResult": ("um_mem.memory.deep_reader", "DeepReadResult"),
    "DependencyIssue": ("um_mem.memory.dependencies", "DependencyIssue"),
    "DependencyValidation": ("um_mem.memory.dependencies", "DependencyValidation"),
    "MemoryClient": ("um_mem.api", "MemoryClient"),
    "MemoryBusyError": ("um_mem.memory.backends.base", "MemoryBusyError"),
    "BackendCapabilities": ("um_mem.memory.backends.contracts", "BackendCapabilities"),
    "PinnedRecordRef": ("um_mem.memory.record_refs", "PinnedRecordRef"),
    "MemoryRetractionReceipt": ("um_mem.memory.mutation", "MemoryRetractionReceipt"),
    "RecallExplanation": ("um_mem.memory.recall_explanation", "RecallExplanation"),
    "RememberResult": ("um_mem.api", "RememberResult"),
    "WorkingRecordRef": ("um_mem.memory.working_set", "WorkingRecordRef"),
    "WorkingRecordSnapshot": ("um_mem.memory.working_set", "WorkingRecordSnapshot"),
    "WorkingSetRead": ("um_mem.memory.working_set", "WorkingSetRead"),
    "WorkingSnapshotBudgetError": ("um_mem.memory.working_set", "WorkingSnapshotBudgetError"),
    "recall": ("um_mem.api", "recall"),
    "remember": ("um_mem.api", "remember"),
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
    "ActivationLease",
    "ActiveMemoryMask",
    "ActiveMemorySelector",
    "AgentSession",
    "BackendCapabilities",
    "DeepReadResult",
    "DependencyIssue",
    "DependencyValidation",
    "MemoryBusyError",
    "MemoryClient",
    "MemoryRetractionReceipt",
    "PinnedRecordRef",
    "RecallExplanation",
    "RememberResult",
    "WorkingRecordRef",
    "WorkingRecordSnapshot",
    "WorkingSetRead",
    "WorkingSnapshotBudgetError",
    "__version__",
    "recall",
    "remember",
]
