"""Fail-closed host and memory-service runtime capability negotiation.

This contract is intentionally separate from backend storage capabilities and
plan/proof capability declarations.  A host adapter owns its manifest and
only marks a capability effective after runtime conformance has passed.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

_CONTRACT_VERSION = 1
_SUPPORTED_PROFILES = frozenset(
    {"deep_harness", "mcp_tools_only", "no_memory", "subagent_light", "workflow"}
)
_DEEP_PREREQUISITES = (
    ("stable_workspace_id", "missing_stable_workspace_id"),
    ("stable_session_id", "missing_stable_session_id"),
    ("pre_turn_delivery", "missing_pre_turn_delivery"),
    ("turn_completion", "missing_turn_completion"),
    ("observed_tool_outcomes", "missing_observed_tool_outcomes"),
)
_OPTIONAL_FEATURES = (
    ("live_state_transitions", "live_state_capture", "live_state_transitions_unavailable"),
    ("compaction_callbacks", "compaction_capture", "compaction_callbacks_unavailable"),
    (
        "prospective_trigger_events",
        "prospective_triggers",
        "prospective_trigger_events_unavailable",
    ),
    (
        "action_authorization_channel",
        "authorized_actions",
        "action_authorization_channel_unavailable",
    ),
)
_HOST_CAPABILITY_NAMES = frozenset(
    {
        "stable_workspace_id",
        "stable_session_id",
        "stable_turn_id",
        "stable_task_id",
        "stable_subtask_id",
        "stable_tool_id",
        "stable_artifact_id",
        "pre_turn_delivery",
        "turn_completion",
        "observed_tool_outcomes",
        "tool_call_events",
        "tool_result_events",
        "tool_failure_events",
        "file_read_events",
        "file_mutation_events",
        "diff_events",
        "test_events",
        "verifier_outcomes",
        "terminal_outcomes",
        "live_state_transitions",
        "compaction_callbacks",
        "prospective_trigger_events",
        "action_authorization_channel",
        "parent_child_lineage",
        "user_identity_channel",
        "user_approval_channel",
        "async_background",
    }
)
_SERVICE_CAPABILITY_NAMES = frozenset(
    {
        "revision_support",
        "tombstone_support",
        "erase_support",
        "index_channels",
        "scoped_events",
        "consolidation",
        "explainability",
    }
)


@dataclass(frozen=True)
class HostCapabilityManifest:
    """Effective lifecycle capability evidence produced inside a trusted host adapter."""

    contract_version: int = _CONTRACT_VERSION
    host_id: str = ""
    adapter_id: str = ""
    transport: str = "unknown"
    conformance_passed: bool = False
    stable_workspace_id: bool = False
    stable_session_id: bool = False
    stable_turn_id: bool = False
    stable_task_id: bool = False
    stable_subtask_id: bool = False
    stable_tool_id: bool = False
    stable_artifact_id: bool = False
    pre_turn_delivery: bool = False
    turn_completion: bool = False
    observed_tool_outcomes: bool = False
    tool_call_events: bool = False
    tool_result_events: bool = False
    tool_failure_events: bool = False
    file_read_events: bool = False
    file_mutation_events: bool = False
    diff_events: bool = False
    test_events: bool = False
    verifier_outcomes: bool = False
    terminal_outcomes: bool = False
    live_state_transitions: bool = False
    compaction_callbacks: bool = False
    prospective_trigger_events: bool = False
    action_authorization_channel: bool = False
    parent_child_lineage: bool = False
    user_identity_channel: bool = False
    user_approval_channel: bool = False
    async_background: bool = False

    def __post_init__(self) -> None:
        if self.contract_version != _CONTRACT_VERSION:
            raise ValueError(f"unsupported host capability contract: {self.contract_version}")

    def supports(self, capability: str) -> bool:
        """Return an effective capability, never an unprobed declaration."""

        if capability not in _HOST_CAPABILITY_NAMES:
            raise ValueError(f"unknown host capability: {capability}")
        return bool(self.conformance_passed and getattr(self, capability))

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "host_id": self.host_id,
            "adapter_id": self.adapter_id,
            "transport": self.transport,
            "conformance_passed": self.conformance_passed,
            "capabilities": {
                capability: self.supports(capability)
                for capability in sorted(_HOST_CAPABILITY_NAMES)
            },
        }


@dataclass(frozen=True)
class MemoryServiceCapabilities:
    """Runtime-probed service facts consumed by host-profile negotiation."""

    contract_version: int = _CONTRACT_VERSION
    conformance_passed: bool = False
    healthy: bool = False
    api_version: int = 0
    schema_version: int = 0
    selected_backend: str = ""
    revision_support: bool = False
    tombstone_support: bool = False
    erase_support: bool = False
    index_channels: bool = False
    scoped_events: bool = False
    consolidation: bool = False
    explainability: bool = False

    def __post_init__(self) -> None:
        if self.contract_version != _CONTRACT_VERSION:
            raise ValueError(f"unsupported service capability contract: {self.contract_version}")
        if self.api_version < 0 or self.schema_version < 0:
            raise ValueError("service API and schema versions must not be negative")

    @property
    def ready(self) -> bool:
        return bool(self.conformance_passed and self.healthy)

    def supports(self, capability: str) -> bool:
        if capability not in _SERVICE_CAPABILITY_NAMES:
            raise ValueError(f"unknown service capability: {capability}")
        return bool(self.ready and getattr(self, capability))

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "conformance_passed": self.conformance_passed,
            "healthy": self.ready,
            "api_version": self.api_version,
            "schema_version": self.schema_version,
            "selected_backend": self.selected_backend,
            "capabilities": {
                capability: self.supports(capability)
                for capability in sorted(_SERVICE_CAPABILITY_NAMES)
            },
        }


@dataclass(frozen=True)
class NegotiatedRuntime:
    """The fail-closed runtime profile fixed for one host/service pairing."""

    contract_version: int = _CONTRACT_VERSION
    requested_profile: str = "deep_harness"
    effective_profile: str = "unavailable"
    integration_mode: str = "unavailable"
    deep_agent_support: bool = False
    enabled_features: tuple[str, ...] = ()
    disabled_reasons: tuple[str, ...] = ()
    budgets: tuple[tuple[str, int], ...] = ()
    host: HostCapabilityManifest = field(default_factory=HostCapabilityManifest)
    service: MemoryServiceCapabilities = field(default_factory=MemoryServiceCapabilities)

    def capability_modes(self) -> dict[str, str]:
        deep = self.effective_profile == "deep_harness"
        available = self.effective_profile != "no_memory" and self.service.ready
        return {
            "capture": "active" if deep and self.host.supports("turn_completion") else "off",
            "working_continuity": (
                "active" if deep and self.host.supports("compaction_callbacks") else "off"
            ),
            "reconstruction": (
                "active" if available and self.service.supports("index_channels") else "off"
            ),
            "consolidation": (
                "active" if available and self.service.supports("consolidation") else "off"
            ),
            "prospective": (
                "active" if deep and self.host.supports("prospective_trigger_events") else "off"
            ),
            "outcome_learning": (
                "active"
                if deep and self.host.supports("verifier_outcomes")
                else "shadow"
                if deep and self.host.supports("observed_tool_outcomes")
                else "off"
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "requested_profile": self.requested_profile,
            "effective_profile": self.effective_profile,
            "integration_mode": self.integration_mode,
            "deep_agent_support": self.deep_agent_support,
            "enabled_features": list(self.enabled_features),
            "capability_modes": self.capability_modes(),
            "disabled_reasons": list(self.disabled_reasons),
            "budgets": dict(self.budgets),
            "health": self.service.ready,
            "host": self.host.to_dict(),
            "service": self.service.to_dict(),
        }


def negotiate_runtime(
    host: HostCapabilityManifest,
    service: MemoryServiceCapabilities | None = None,
    *,
    requested_profile: str = "deep_harness",
    budgets: Mapping[str, int] | None = None,
) -> NegotiatedRuntime:
    """Negotiate one explicit profile without silently changing its meaning."""

    service = service or MemoryServiceCapabilities()
    requested = str(requested_profile or "").strip().lower()
    normalized_budgets = _normalize_budgets(budgets)
    disabled = _service_disabled_reasons(service)
    enabled: list[str] = []

    for capability, feature, reason in _OPTIONAL_FEATURES:
        if host.supports(capability):
            enabled.append(feature)
        else:
            disabled.append(reason)

    if requested == "no_memory":
        return _runtime(
            requested=requested,
            effective="no_memory",
            mode="no_memory",
            enabled=(),
            disabled=disabled,
            budgets=normalized_budgets,
            host=host,
            service=service,
        )

    if requested == "mcp_tools_only":
        if service.ready:
            enabled.insert(0, "routed_memory_tools")
        disabled.insert(0, "mcp_transport_has_no_host_lifecycle_capture")
        return _runtime(
            requested=requested,
            effective="mcp_tools_only",
            mode="mcp_tools_only",
            enabled=enabled,
            disabled=disabled,
            budgets=normalized_budgets,
            host=host,
            service=service,
        )

    if requested != "deep_harness":
        disabled.insert(0, _unsupported_profile_reason(requested))
        return _runtime(
            requested=requested or "unknown",
            effective="unavailable",
            mode="unavailable",
            enabled=(),
            disabled=disabled,
            budgets=normalized_budgets,
            host=host,
            service=service,
        )

    missing_deep = [
        reason for capability, reason in _DEEP_PREREQUISITES if not host.supports(capability)
    ]
    disabled.extend(missing_deep)
    if not service.ready or missing_deep:
        return _runtime(
            requested=requested,
            effective="unavailable",
            mode="native_lifecycle_unqualified",
            enabled=(),
            disabled=disabled,
            budgets=normalized_budgets,
            host=host,
            service=service,
        )

    enabled = [
        "stable_host_identity",
        "pre_turn_context_delivery",
        "turn_completion_capture",
        "observed_tool_outcomes",
        *enabled,
    ]
    return _runtime(
        requested=requested,
        effective="deep_harness",
        mode="deep_harness",
        enabled=enabled,
        disabled=disabled,
        budgets=normalized_budgets,
        host=host,
        service=service,
    )


def _service_disabled_reasons(service: MemoryServiceCapabilities) -> list[str]:
    if service.ready:
        return []
    if not service.conformance_passed:
        return ["memory_service_conformance_unconfirmed"]
    return ["memory_service_unhealthy"]


def _unsupported_profile_reason(requested_profile: str) -> str:
    if requested_profile in _SUPPORTED_PROFILES:
        return f"profile_not_implemented:{requested_profile}"
    return f"unsupported_requested_profile:{requested_profile or 'empty'}"


def _normalize_budgets(budgets: Mapping[str, int] | None) -> tuple[tuple[str, int], ...]:
    if budgets is None:
        return ()
    normalized: list[tuple[str, int]] = []
    for name, value in budgets.items():
        key = str(name).strip()
        if not key or isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "runtime budgets must have non-empty names and non-negative integer values"
            )
        normalized.append((key, value))
    return tuple(sorted(normalized))


def _runtime(
    *,
    requested: str,
    effective: str,
    mode: str,
    enabled: list[str] | tuple[str, ...],
    disabled: list[str],
    budgets: tuple[tuple[str, int], ...],
    host: HostCapabilityManifest,
    service: MemoryServiceCapabilities,
) -> NegotiatedRuntime:
    return NegotiatedRuntime(
        requested_profile=requested,
        effective_profile=effective,
        integration_mode=mode,
        deep_agent_support=effective == "deep_harness",
        enabled_features=tuple(dict.fromkeys(enabled)),
        disabled_reasons=tuple(dict.fromkeys(disabled)),
        budgets=budgets,
        host=host,
        service=service,
    )
