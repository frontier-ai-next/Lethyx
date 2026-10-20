import json

from lethyx.integrations.hermes_provider import UmMemMemoryProvider


def test_hermes_exposes_free_topology_and_active_mask_contracts(monkeypatch) -> None:
    provider = UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "backend": "mempalace",
        "repo_root": "unused",
    }
    provider._session_id = "session-1"
    calls: list[tuple[str, dict[str, object]]] = []
    caller_roles: list[str] = []

    def call(
        tool: str,
        args: dict[str, object],
        *,
        caller_role: str = "parent",
    ) -> dict[str, object]:
        calls.append((tool, args))
        caller_roles.append(caller_role)
        if tool == "block_map":
            return {
                "ok": True,
                "roots": [
                    {
                        "block_id": "root-id",
                        "name": "Workspace memory",
                        "synopsis": "Protected profile root.",
                        "child_ids": [],
                        "protected": True,
                    }
                ],
            }
        if tool == "resolve_active_mask":
            return {
                "ok": True,
                "lease": {
                    "rendered_context": "active context",
                    "trace": {
                        "event_type": "active_resolve",
                        "mask_id": "mask-1",
                        "resolution_status": "resolved",
                    },
                },
            }
        return {"ok": True, **args}

    monkeypatch.setattr(provider, "_call", call)
    monkeypatch.setattr(
        provider, "_remember_and_export", lambda payload, **kwargs: call(kwargs["tool"], payload)
    )

    schemas = {schema["name"]: schema for schema in provider.get_tool_schemas()}
    required = {
        "lethyx_provider_create_block",
        "lethyx_provider_amend_block",
        "lethyx_provider_move_block",
        "lethyx_provider_archive_block",
        "lethyx_provider_revive_block",
        "lethyx_provider_delete_block",
        "lethyx_provider_split_block",
        "lethyx_provider_merge_blocks",
        "lethyx_provider_add_block_edge",
        "lethyx_provider_remove_block_edge",
        "lethyx_provider_get_active_mask",
        "lethyx_provider_set_active_mask",
        "lethyx_provider_amend_active_mask",
        "lethyx_provider_transition_active_mask",
        "lethyx_provider_resolve_active_mask",
        "lethyx_provider_navigation_envelope",
        "lethyx_provider_observe_maintenance",
        "lethyx_provider_consolidate_observations",
    }
    assert required <= schemas.keys()
    create = schemas["lethyx_provider_create_block"]["parameters"]
    assert "block_id" not in create["properties"]
    assert set(create["required"]) == {"name", "synopsis", "parent_id", "reason"}
    amend = schemas["lethyx_provider_amend_block"]["parameters"]
    assert amend["properties"]["expected_revision"]["type"] == "integer"
    assert "parent_id" not in amend["properties"]
    assert "archived" not in amend["properties"]
    set_mask = schemas["lethyx_provider_set_active_mask"]["parameters"]
    assert set(set_mask["required"]) == {"selected_block_ids", "activation_reason"}
    assert set_mask["properties"]["token_budget"]["default"] == 2048

    created = json.loads(
        provider.handle_tool_call(
            "lethyx_provider_create_block",
            {
                "name": "Research findings",
                "synopsis": "Verified research evidence.",
                "parent_id": "root-id",
                "reason": "new durable subject",
            },
        )
    )
    assert created["ok"] is True
    assert calls[-1] == (
        "create_block",
        {
            "name": "Research findings",
            "synopsis": "Verified research evidence.",
            "parent_id": "root-id",
            "reason": "new durable subject",
        },
    )

    navigated = json.loads(
        provider.handle_tool_call(
            "lethyx_provider_navigation_envelope",
            {"block_id": "root-id"},
        )
    )
    assert navigated["ok"] is True
    assert calls[-1] == ("navigation_envelope", {"block_id": "root-id"})

    consolidated = json.loads(
        provider.handle_tool_call(
            "lethyx_provider_consolidate_observations",
            {
                "source_ids": ["source-1"],
                "content": "Bounded derived view.",
                "title": "Derived view",
                "reason": "explicit consolidation",
                "verification_reports": [{"report_id": "report-1"}],
                "semantic_verification_report": {"report_id": "semantic-report-1"},
                "requested_block_id": "root-id",
            },
        )
    )
    assert consolidated["ok"] is False
    assert consolidated["error"] == "host_authority_required"
    assert calls[-1][0] == "navigation_envelope"
    assert set(caller_roles) == {"parent"}

    selected = json.loads(
        provider.handle_tool_call(
            "lethyx_provider_set_active_mask",
            {
                "selected_block_ids": ["root-id"],
                "activation_reason": "keep the current research context across rounds",
            },
        )
    )
    assert selected["ok"] is True
    assert calls[-1] == (
        "set_active_mask",
        {
            "selected_block_ids": ["root-id"],
            "activation_reason": "keep the current research context across rounds",
            "token_budget": 2048,
        },
    )

    prefetched = json.loads(provider.prefetch("ignored", session_id="session-1"))
    assert [tool for tool, _args in calls[-2:]] == ["block_map", "resolve_active_mask"]
    assert set(prefetched) == {"lethyx_navigation", "lethyx_active", "next"}
    assert prefetched["lethyx_navigation"]["roots"][0]["block_id"] == "root-id"
    assert prefetched["lethyx_active"]["context"] == "active context"
    assert prefetched["lethyx_active"]["status"] == "resolved"
    assert prefetched["lethyx_active"]["mask_id"] == "mask-1"
    assert "trace" not in prefetched["lethyx_active"]
    assert "injected automatically" in prefetched["lethyx_active"]["guidance"]
    assert "amend" in prefetched["lethyx_active"]["guidance"].casefold()
    prompt = provider.system_prompt_block()
    assert "recall is temporary" in prompt
    assert "set or amend the active mask" in prompt
    assert "suspend or deactivate" in prompt.casefold()


def test_hermes_prefetch_exposes_active_mask_resolution_failure(monkeypatch) -> None:
    provider = UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "backend": "mempalace",
        "repo_root": "unused",
    }

    def call(tool: str, _args: dict[str, object], **_kwargs: object) -> dict[str, object]:
        if tool == "block_map":
            return {"ok": True, "roots": []}
        return {"ok": False, "error": "service unavailable"}

    monkeypatch.setattr(provider, "_call", call)
    monkeypatch.setattr(
        "lethyx.integrations.hermes_provider.record_memory_event",
        lambda *args, **kwargs: None,
    )

    prefetched = json.loads(provider.prefetch("ignored", session_id="session-1"))

    assert prefetched["lethyx_active"] == {
        "status": "unavailable",
        "context": "",
        "error": "active_mask_resolution_failed",
        "guidance": (
            "Active-memory delivery is temporarily unavailable. Continue the task; memory is "
            "advisory and must never block agent work."
        ),
    }


def test_hermes_preserves_explicit_local_service_lifecycle_owner(monkeypatch) -> None:
    provider = UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "profile_id": "profile-1",
        "backend": "mempalace",
        "repo_root": "unused",
    }
    monkeypatch.setenv("LETHYX_LOCAL_SERVICE_OWNER_PID", "12345")
    monkeypatch.setenv("LETHYX_LOCAL_SERVICE_IDLE_SECONDS", "7200")
    monkeypatch.setenv("LETHYX_UNTRUSTED_RUNTIME_OVERRIDE", "must-not-pass")

    environment = provider._tool_env()

    assert environment["LETHYX_LOCAL_SERVICE_OWNER_PID"] == "12345"
    assert environment["LETHYX_LOCAL_SERVICE_IDLE_SECONDS"] == "7200"
    assert "LETHYX_UNTRUSTED_RUNTIME_OVERRIDE" not in environment
