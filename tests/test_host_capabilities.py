from dataclasses import replace

import pytest

from um_mem.integrations import (
    HostCapabilityManifest,
    MemoryServiceCapabilities,
    negotiate_runtime,
)


def _ready_service() -> MemoryServiceCapabilities:
    return MemoryServiceCapabilities(
        conformance_passed=True,
        healthy=True,
        api_version=1,
        schema_version=1,
        selected_backend="mempalace",
    )


def _verified_deep_host(**overrides: bool) -> HostCapabilityManifest:
    values: dict[str, bool] = {
        "conformance_passed": True,
        "stable_workspace_id": True,
        "stable_session_id": True,
        "pre_turn_delivery": True,
        "turn_completion": True,
        "observed_tool_outcomes": True,
        "live_state_transitions": True,
        "compaction_callbacks": True,
        "prospective_trigger_events": True,
        "action_authorization_channel": True,
    }
    values.update(overrides)
    return HostCapabilityManifest(host_id="test-host", adapter_id="test-adapter", **values)


def test_host_capabilities_default_to_false_and_deep_fails_closed() -> None:
    host = HostCapabilityManifest(stable_workspace_id=True)

    runtime = negotiate_runtime(host, _ready_service())

    assert host.supports("stable_workspace_id") is False
    assert host.to_dict()["capabilities"]["stable_workspace_id"] is False
    assert runtime.effective_profile == "unavailable"
    assert runtime.deep_agent_support is False
    assert {
        "missing_stable_workspace_id",
        "missing_stable_session_id",
        "missing_pre_turn_delivery",
        "missing_turn_completion",
        "missing_observed_tool_outcomes",
    }.issubset(runtime.disabled_reasons)


@pytest.mark.parametrize(
    ("capability", "reason"),
    [
        ("stable_workspace_id", "missing_stable_workspace_id"),
        ("stable_session_id", "missing_stable_session_id"),
        ("pre_turn_delivery", "missing_pre_turn_delivery"),
        ("turn_completion", "missing_turn_completion"),
        ("observed_tool_outcomes", "missing_observed_tool_outcomes"),
    ],
)
def test_every_deep_prerequisite_is_individually_fail_closed(
    capability: str,
    reason: str,
) -> None:
    runtime = negotiate_runtime(
        replace(_verified_deep_host(), **{capability: False}), _ready_service()
    )

    assert runtime.effective_profile == "unavailable"
    assert reason in runtime.disabled_reasons


@pytest.mark.parametrize(
    ("capability", "feature", "reason"),
    [
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
    ],
)
def test_optional_host_capabilities_limit_only_their_feature(
    capability: str,
    feature: str,
    reason: str,
) -> None:
    runtime = negotiate_runtime(
        replace(_verified_deep_host(), **{capability: False}), _ready_service()
    )

    assert runtime.effective_profile == "deep_harness"
    assert feature not in runtime.enabled_features
    assert reason in runtime.disabled_reasons


def test_verified_host_and_service_negotiate_deep_with_budgets() -> None:
    runtime = negotiate_runtime(
        _verified_deep_host(),
        _ready_service(),
        budgets={"context_tokens": 2048, "events": 128},
    )

    assert runtime.effective_profile == "deep_harness"
    assert runtime.deep_agent_support is True
    assert set(runtime.enabled_features) >= {
        "stable_host_identity",
        "pre_turn_context_delivery",
        "turn_completion_capture",
        "observed_tool_outcomes",
        "live_state_capture",
        "compaction_capture",
        "prospective_triggers",
        "authorized_actions",
    }
    assert runtime.to_dict()["budgets"] == {"context_tokens": 2048, "events": 128}


def test_lifecycle_capabilities_publish_off_shadow_or_active_modes() -> None:
    runtime = negotiate_runtime(
        replace(_verified_deep_host(), verifier_outcomes=False),
        replace(_ready_service(), index_channels=True, consolidation=True),
    )

    modes = runtime.to_dict()["capability_modes"]
    assert set(modes.values()) <= {"off", "shadow", "active"}
    assert modes["capture"] == "active"
    assert modes["working_continuity"] == "active"
    assert modes["reconstruction"] == "active"
    assert modes["consolidation"] == "active"
    assert modes["prospective"] == "active"
    assert modes["outcome_learning"] == "shadow"


def test_explicit_mcp_tools_only_never_relabels_itself_as_deep() -> None:
    runtime = negotiate_runtime(
        HostCapabilityManifest(
            host_id="mcp",
            adapter_id="test-mcp",
            transport="mcp",
            conformance_passed=True,
        ),
        _ready_service(),
        requested_profile="mcp_tools_only",
    )

    assert runtime.requested_profile == "mcp_tools_only"
    assert runtime.effective_profile == "mcp_tools_only"
    assert runtime.deep_agent_support is False
    assert "routed_memory_tools" in runtime.enabled_features
    assert "mcp_transport_has_no_host_lifecycle_capture" in runtime.disabled_reasons


def test_no_memory_profile_is_explicit_and_never_an_unhealthy_fallback() -> None:
    runtime = negotiate_runtime(
        HostCapabilityManifest(conformance_passed=True),
        MemoryServiceCapabilities(conformance_passed=True, healthy=False),
        requested_profile="no_memory",
    )

    assert runtime.requested_profile == "no_memory"
    assert runtime.effective_profile == "no_memory"
    assert runtime.integration_mode == "no_memory"
