"""Minimal contracts facade for the Lethyx extraction snapshot.

The original Lethyx facade imports the full phase/control-plane contract
stack. Lethyx keeps the full copied contract modules for porting reference, but
memory imports only need the small typed/evidence surface exported here.
"""

from lethyx.contracts.models import (
    ContractBundle,
    ContractIssue,
    EvidenceRef,
    TrustLevel,
    VerificationReportRef,
    WorkspaceContext,
    json_ready,
)
from lethyx.contracts.validators import ContractValidator

try:
    from lethyx.contracts.evidence import EvidenceResolver, build_workspace_context
except ModuleNotFoundError:  # pragma: no cover - only until support modules are ported.
    EvidenceResolver = None  # type: ignore[assignment]
    build_workspace_context = None  # type: ignore[assignment]

__all__ = [
    "ContractBundle",
    "ContractIssue",
    "ContractValidator",
    "EvidenceRef",
    "EvidenceResolver",
    "TrustLevel",
    "VerificationReportRef",
    "WorkspaceContext",
    "build_workspace_context",
    "json_ready",
]
