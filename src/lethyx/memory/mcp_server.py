"""Lethyx MCP server.

Run with:

    PYTHONPATH=src python -m lethyx.memory.mcp_server

Hermes will expose these tools as `mcp_<server_name>_<tool_name>`.
"""

import os
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from lethyx.contracts.memory import MemoryScope
from lethyx.memory.topology_schema import BLOCK_EVOLUTION_TARGET_SCHEMA, BLOCK_MERGE_SOURCE_SCHEMA
from lethyx.memory.local_service import (
    LocalServiceClientPool,
    build_local_service_identity,
    local_service_scope_lineage,
)
from lethyx.memory.recall_plan import (
    MEMORY_READ_GUIDANCE,
    MEMORY_WRITE_GUIDANCE,
    RetrievalMode,
    SourceHorizon,
)

TOOL_NAMES = (
    "health",
    "block_map",
    "expand_block",
    "navigation_envelope",
    "create_block",
    "amend_block",
    "move_block",
    "archive_block",
    "revive_block",
    "delete_block",
    "split_block",
    "merge_blocks",
    "pending_topology_evolutions",
    "resume_topology_evolution",
    "abort_topology_evolution",
    "add_block_edge",
    "remove_block_edge",
    "get_active_mask",
    "set_active_mask",
    "amend_active_mask",
    "transition_active_mask",
    "resolve_active_mask",
    "observe_maintenance",
    "consolidate_observations",
    "remember",
    "amend",
    "recall",
    "read_exact_record",
    "read_exact_records",
    "list_records",
    "operation_status",
    "projection_barrier",
    "read_working_set",
    "build_context",
    "forget",
    "add_edge",
    "export_graph",
    "export_obsidian",
)

_LOCAL_SERVICE_CLIENTS = LocalServiceClientPool()


def close_local_service_clients() -> None:
    _LOCAL_SERVICE_CLIENTS.close()


def _tool_timeout() -> int:
    try:
        return max(1, int(os.getenv("LETHYX_MCP_TOOL_TIMEOUT_SECONDS", "120")))
    except ValueError:
        return 120


def _export_path_is_authorized(identity, arguments: dict[str, Any]) -> bool:
    vault_dir = arguments.get("vault_dir")
    if vault_dir in (None, ""):
        return True
    if not isinstance(vault_dir, str):
        return False
    configured_vault = os.getenv("LETHYX_OBSIDIAN_VAULT", "").strip()
    expected = (
        Path(configured_vault) if configured_vault else identity.repo_root / "exports" / "obsidian"
    )
    try:
        return Path(vault_dir).expanduser().resolve() == expected.expanduser().resolve()
    except (OSError, RuntimeError):
        return False


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a Lethyx tool through its identity-bound shared listener."""

    if name == "remember":
        operation_id = arguments.get("operation_id", "")
        if (
            not isinstance(operation_id, str)
            or operation_id != operation_id.strip()
            or len(operation_id) > 256
        ):
            return {"ok": False, "error": "operation_id must be a bounded exact string"}
        arguments = {**arguments, "operation_id": operation_id or uuid4().hex}
    try:
        identity = build_local_service_identity(
            repo_root=None,
            workspace_id=None,
            backend=None,
            caller_role="operator",
        )
        claimed_identity = build_local_service_identity(
            repo_root=(
                identity.repo_root
                if arguments.get("repo_root") in (None, "")
                else arguments.get("repo_root")
            ),
            workspace_id=(
                identity.workspace_id
                if arguments.get("workspace_id") in (None, "")
                else arguments.get("workspace_id")
            ),
            profile_id=identity.profile_id,
            backend=(
                identity.backend
                if arguments.get("backend") in (None, "")
                else arguments.get("backend")
            ),
            caller_role="operator",
        )
    except ValueError:
        return {"ok": False, "error": "local service identity rejected"}
    if claimed_identity != identity:
        return {"ok": False, "error": "local service request identity is not authorized"}
    if name == "export_obsidian" and not _export_path_is_authorized(identity, arguments):
        return {"ok": False, "error": "local service export path is not authorized"}
    try:
        scope_lineage = local_service_scope_lineage(
            identity,
            session_id=os.getenv("LETHYX_SESSION_ID", "default"),
            project_id=os.getenv("LETHYX_PROJECT_ID", ""),
            task_id=os.getenv("LETHYX_TASK_ID", ""),
            subtask_id=os.getenv("LETHYX_SUBTASK_ID", ""),
            user_id=os.getenv("LETHYX_USER_ID", ""),
        )
    except ValueError:
        return {"ok": False, "error": "local service scope lineage rejected"}
    try:
        result = _LOCAL_SERVICE_CLIENTS.call(
            identity,
            name,
            arguments,
            timeout_seconds=_tool_timeout(),
            session_id=scope_lineage.session_id,
            scope_lineage=scope_lineage,
            mutation_correlation_id=(
                arguments.get("operation_id", "") if name == "remember" else ""
            ),
        )
        if name == "remember":
            result.setdefault("operation_id", arguments["operation_id"])
        if name == "health":
            service = _LOCAL_SERVICE_CLIENTS.service_status(identity)
            if "runtime" not in result:
                result["integration_mode"] = "mcp_tools_only"
                result["deep_agent_support"] = False
                result["degraded_reasons"] = ["mcp_transport_has_no_host_lifecycle_capture"]
                result["runtime"] = {
                    "requested_profile": "mcp_tools_only",
                    "effective_profile": "mcp_tools_only",
                    "integration_mode": "mcp_tools_only",
                    "deep_agent_support": False,
                    "health": service.get("readiness") == "ready",
                }
            result["service"] = service
        return result
    except ValueError:
        return {"ok": False, "error": "local service identity rejected"}


def _make_tool_arguments_strict(server: FastMCP) -> None:
    for name in TOOL_NAMES:
        tool = server._tool_manager.get_tool(name)
        if tool is None:
            raise RuntimeError(f"missing required Lethyx MCP tool: {name}")
        argument_model = tool.fn_metadata.arg_model
        argument_model.model_config["extra"] = "forbid"
        argument_model.model_rebuild(force=True)
        tool.parameters = argument_model.model_json_schema(by_alias=True)


def create_mcp_server() -> FastMCP:
    """Create the stdio MCP server with project-scoped Lethyx tools."""

    server = FastMCP("lethyx", instructions=MEMORY_READ_GUIDANCE + " " + MEMORY_WRITE_GUIDANCE)

    @server.tool()
    def health() -> dict[str, Any]:
        """Check the selected Lethyx backend for a workspace."""

        return _call_tool("health", {})

    @server.tool()
    def block_map() -> dict[str, Any]:
        """Show the small root map without exposing memory record contents."""

        return _call_tool("block_map", {})

    @server.tool()
    def expand_block(block_id: str) -> dict[str, Any]:
        """Expand any logical block reachable from the root map."""

        return _call_tool("expand_block", {"block_id": block_id})

    @server.tool()
    def navigation_envelope(block_id: str) -> dict[str, Any]:
        """Return tree, graph, health, and advisory maintenance facts for one block."""

        return _call_tool("navigation_envelope", {"block_id": block_id})

    @server.tool()
    def create_block(
        name: str,
        synopsis: str,
        parent_id: str,
        reason: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        scope: str = "workspace",
        index_only: bool = False,
        token_budget: int = 1000,
        expires_at: str = "",
    ) -> dict[str, Any]:
        """Create one workspace-owned logical memory block."""

        return _call_tool(
            "create_block",
            {
                "name": name,
                "synopsis": synopsis,
                "parent_id": parent_id,
                "reason": reason,
                "tags": tags,
                "metadata": metadata,
                "scope": scope,
                "index_only": index_only,
                "token_budget": token_budget,
                "expires_at": expires_at,
            },
        )

    @server.tool()
    def amend_block(
        block_id: str,
        expected_revision: int,
        reason: str,
        name: str | None = None,
        synopsis: str | None = None,
        tags: list[str] | None = None,
        metadata_patch: dict[str, Any] | None = None,
        token_budget: int | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        """CAS-amend or revive one workspace-owned logical memory block."""

        return _call_tool(
            "amend_block",
            {
                "block_id": block_id,
                "expected_revision": expected_revision,
                "reason": reason,
                "name": name,
                "synopsis": synopsis,
                "tags": tags,
                "metadata_patch": metadata_patch,
                "token_budget": token_budget,
                "expires_at": expires_at,
            },
        )

    @server.tool()
    def archive_block(
        block_id: str,
        expected_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        """Archive one workspace-owned logical memory block."""

        return _call_tool(
            "archive_block",
            {
                "block_id": block_id,
                "expected_revision": expected_revision,
                "reason": reason,
            },
        )

    @server.tool()
    def move_block(
        block_id: str,
        parent_id: str,
        expected_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        return _call_tool(
            "move_block",
            {
                "block_id": block_id,
                "parent_id": parent_id,
                "expected_revision": expected_revision,
                "reason": reason,
            },
        )

    @server.tool()
    def revive_block(
        block_id: str,
        expected_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        return _call_tool(
            "revive_block",
            {
                "block_id": block_id,
                "expected_revision": expected_revision,
                "reason": reason,
            },
        )

    @server.tool()
    def delete_block(
        block_id: str,
        expected_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        return _call_tool(
            "delete_block",
            {
                "block_id": block_id,
                "expected_revision": expected_revision,
                "reason": reason,
            },
        )

    @server.tool()
    def split_block(
        block_id: str,
        expected_revision: int,
        targets: Annotated[
            list[dict[str, Any]], Field(json_schema_extra={"items": BLOCK_EVOLUTION_TARGET_SCHEMA})
        ],
        reason: str,
        operation_id: str = "",
    ) -> dict[str, Any]:
        return _call_tool(
            "split_block",
            {
                "block_id": block_id,
                "expected_revision": expected_revision,
                "targets": targets,
                "reason": reason,
                "operation_id": operation_id,
            },
        )

    @server.tool()
    def merge_blocks(
        sources: Annotated[
            list[dict[str, Any]], Field(json_schema_extra={"items": BLOCK_MERGE_SOURCE_SCHEMA})
        ],
        target: Annotated[dict[str, Any], Field(json_schema_extra=BLOCK_EVOLUTION_TARGET_SCHEMA)],
        parent_id: str,
        reason: str,
        operation_id: str = "",
    ) -> dict[str, Any]:
        return _call_tool(
            "merge_blocks",
            {
                "sources": sources,
                "target": target,
                "parent_id": parent_id,
                "reason": reason,
                "operation_id": operation_id,
            },
        )

    @server.tool()
    def pending_topology_evolutions() -> dict[str, Any]:
        return _call_tool("pending_topology_evolutions", {})

    @server.tool()
    def resume_topology_evolution(operation_id: str) -> dict[str, Any]:
        return _call_tool("resume_topology_evolution", {"operation_id": operation_id})

    @server.tool()
    def abort_topology_evolution(operation_id: str, reason: str) -> dict[str, Any]:
        return _call_tool(
            "abort_topology_evolution", {"operation_id": operation_id, "reason": reason}
        )

    @server.tool()
    def add_block_edge(
        src_id: str,
        dst_id: str,
        edge_type: str,
        reason: str,
    ) -> dict[str, Any]:
        return _call_tool(
            "add_block_edge",
            {
                "src_id": src_id,
                "dst_id": dst_id,
                "edge_type": edge_type,
                "reason": reason,
            },
        )

    @server.tool()
    def remove_block_edge(
        edge_id: str,
        expected_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        return _call_tool(
            "remove_block_edge",
            {
                "edge_id": edge_id,
                "expected_revision": expected_revision,
                "reason": reason,
            },
        )

    @server.tool()
    def get_active_mask(mask_id: str = "") -> dict[str, Any]:
        return _call_tool("get_active_mask", {"mask_id": mask_id})

    @server.tool()
    def set_active_mask(
        selected_block_ids: list[str],
        token_budget: int,
        activation_reason: str,
        context_refs: list[str] | None = None,
        pinned_record_ids: list[str] | None = None,
        selectors: list[dict[str, Any]] | None = None,
        priorities: list[dict[str, Any]] | None = None,
        block_token_budgets: list[dict[str, Any]] | None = None,
        expires_at: str = "",
    ) -> dict[str, Any]:
        return _call_tool(
            "set_active_mask",
            {
                "selected_block_ids": selected_block_ids,
                "token_budget": token_budget,
                "activation_reason": activation_reason,
                "context_refs": context_refs,
                "pinned_record_ids": pinned_record_ids,
                "selectors": selectors,
                "priorities": priorities,
                "block_token_budgets": block_token_budgets,
                "expires_at": expires_at,
            },
        )

    @server.tool()
    def amend_active_mask(
        mask_id: str,
        expected_revision: int,
        activation_reason: str,
        selected_block_ids: list[str] | None = None,
        context_refs: list[str] | None = None,
        pinned_record_ids: list[str] | None = None,
        selectors: list[dict[str, Any]] | None = None,
        priorities: list[dict[str, Any]] | None = None,
        token_budget: int | None = None,
        block_token_budgets: list[dict[str, Any]] | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        return _call_tool(
            "amend_active_mask",
            {
                "mask_id": mask_id,
                "expected_revision": expected_revision,
                "activation_reason": activation_reason,
                "selected_block_ids": selected_block_ids,
                "context_refs": context_refs,
                "pinned_record_ids": pinned_record_ids,
                "selectors": selectors,
                "priorities": priorities,
                "token_budget": token_budget,
                "block_token_budgets": block_token_budgets,
                "expires_at": expires_at,
            },
        )

    @server.tool()
    def transition_active_mask(
        mask_id: str,
        expected_revision: int,
        transition: str,
        reason: str,
    ) -> dict[str, Any]:
        return _call_tool(
            "transition_active_mask",
            {
                "mask_id": mask_id,
                "expected_revision": expected_revision,
                "transition": transition,
                "reason": reason,
            },
        )

    @server.tool()
    def resolve_active_mask(
        mask_id: str = "",
        expected_revision: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        return _call_tool(
            "resolve_active_mask",
            {
                "mask_id": mask_id,
                "expected_revision": expected_revision,
                "max_tokens": max_tokens,
            },
        )

    @server.tool()
    def observe_maintenance(
        block_id: str,
        observation_type: str,
        values: dict[str, int | float],
        threshold_revision: str,
        evidence_ids: list[str],
    ) -> dict[str, Any]:
        """Record advisory maintenance evidence without mutating the topology."""

        return _call_tool(
            "observe_maintenance",
            {
                "block_id": block_id,
                "observation_type": observation_type,
                "values": values,
                "threshold_revision": threshold_revision,
                "evidence_ids": evidence_ids,
            },
        )

    @server.tool()
    def consolidate_observations(
        source_ids: list[str],
        content: str,
        title: str,
        reason: str,
        verification_reports: list[Any],
        semantic_verification_report: dict[str, Any],
        requested_block_id: str,
    ) -> dict[str, Any]:
        """Run one bounded, evidence-gated consolidation into an explicit leaf."""

        return _call_tool(
            "consolidate_observations",
            {
                "source_ids": source_ids,
                "content": content,
                "title": title,
                "reason": reason,
                "verification_reports": verification_reports,
                "semantic_verification_report": semantic_verification_report,
                "requested_block_id": requested_block_id,
            },
        )

    @server.tool()
    def remember(
        content: str,
        requested_block_id: str,
        operation: str = "add",
        memory_id: str = "",
        repair_of: str = "",
        reason: str = "correction",
        title: str = "",
        kind: str = "observation",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        evidence_kind: str = "",
        evidence_refs: list[Any] | None = None,
        write_mode: str = "standard",
        source_segments: list[dict[str, Any]] | None = None,
        operation_id: str = "",
        source_id: str = "",
        sensitivity: str = "internal",
        content_mode: str = "full",
        operational_risk: str = "low",
        child_claims: list[dict[str, object]] | None = None,
        artifact_id: str = "",
        artifact_revision: str = "",
        valid_from: str = "",
        valid_to: str = "",
        repository_id: str = "",
        checkout: str = "",
        worktree: str = "",
        branch: str = "",
        revision: str = "",
        dirty_state: str = "",
        hard_policy_version: int = 1,
        semantic_catalog_version: str = "",
    ) -> dict[str, Any]:
        """Add, repair, or correct evidence inside one agent-selected logical block."""

        return _call_tool(
            "remember",
            {
                "content": content,
                "operation": operation,
                "memory_id": memory_id,
                "repair_of": repair_of,
                "reason": reason,
                "title": title,
                "kind": kind,
                "tags": tags,
                "metadata": metadata,
                "evidence_kind": evidence_kind,
                "evidence_refs": evidence_refs,
                "write_mode": write_mode,
                "source_segments": source_segments,
                "operation_id": operation_id,
                "requested_block_id": requested_block_id,
                "source_id": source_id,
                "sensitivity": sensitivity,
                "content_mode": content_mode,
                "operational_risk": operational_risk,
                "child_claims": child_claims,
                "artifact_id": artifact_id,
                "artifact_revision": artifact_revision,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "repository_id": repository_id,
                "checkout": checkout,
                "worktree": worktree,
                "branch": branch,
                "revision": revision,
                "dirty_state": dirty_state,
                "hard_policy_version": hard_policy_version,
                "semantic_catalog_version": semantic_catalog_version,
            },
        )

    @server.tool()
    def amend(
        memory_id: str,
        expected_revision: str,
        reason: str,
        content: str | None = None,
        title: str | None = None,
        tags: list[str] | None = None,
        metadata_patch: dict[str, Any] | None = None,
        retention: str | None = None,
        expires_at: str | None = None,
        evidence_refs: list[Any] | None = None,
        requested_block_id: str = "",
    ) -> dict[str, Any]:
        """Append one immutable CAS-guarded advisory-memory revision."""

        return _call_tool(
            "amend",
            {
                "memory_id": memory_id,
                "expected_revision": expected_revision,
                "reason": reason,
                "content": content,
                "title": title,
                "tags": tags,
                "metadata_patch": metadata_patch,
                "retention": retention,
                "expires_at": expires_at,
                "evidence_refs": evidence_refs,
                "requested_block_id": requested_block_id,
            },
        )

    @server.tool()
    def recall(
        query: str,
        limit: int = 4,
        tags: list[str] | None = None,
        source_horizon: SourceHorizon = "durable",
        include_candidates: bool = False,
        logical_block_ids: list[str] | None = None,
        search_all: bool = False,
        retrieval_mode: RetrievalMode = "semantic",
        record_ids: list[str] | None = None,
        max_tokens: int = 2048,
        source_speakers: list[str] | None = None,
        temporal_mode: str = "current",
        as_of: str = "",
        known_as_of: str = "",
        revision_id: str = "",
    ) -> dict[str, Any]:
        """Recall a small relevant set from inspected leaves; raise limit only if insufficient."""

        return _call_tool(
            "recall",
            {
                "query": query,
                "limit": limit,
                "tags": tags,
                "source_horizon": source_horizon,
                "include_candidates": include_candidates,
                "logical_block_ids": logical_block_ids,
                "search_all": search_all,
                "retrieval_mode": retrieval_mode,
                "record_ids": record_ids,
                "max_tokens": max_tokens,
                "source_speakers": source_speakers,
                "temporal_mode": temporal_mode,
                "as_of": as_of,
                "known_as_of": known_as_of,
                "revision_id": revision_id,
            },
        )

    @server.tool()
    def read_exact_record(record_id: str = "", evidence_ref: str = "") -> dict[str, Any]:
        """Read one canonical record by ID or an unambiguous exact evidence reference."""

        return _call_tool(
            "read_exact_record", {"record_id": record_id, "evidence_ref": evidence_ref}
        )

    @server.tool()
    def read_exact_records(record_ids: list[str]) -> dict[str, Any]:
        """Read at most 128 distinct canonical record IDs in caller order."""

        return _call_tool("read_exact_records", {"record_ids": record_ids})

    @server.tool()
    def list_records(
        scope: MemoryScope,
        scope_id: str,
        logical_block_id: str = "",
        limit: int = 32,
        cursor: str = "",
    ) -> dict[str, Any]:
        """List bounded canonical IDs in one explicit scope with a frozen cursor."""

        return _call_tool(
            "list_records",
            {
                "scope": scope,
                "scope_id": scope_id,
                "logical_block_id": logical_block_id,
                "limit": limit,
                "cursor": cursor,
            },
        )

    @server.tool()
    def operation_status(operation_id: str) -> dict[str, Any]:
        """Read the original operation receipt and current projection status."""

        return _call_tool("operation_status", {"operation_id": operation_id})

    @server.tool()
    def projection_barrier(
        record_id: str, commit_revision: int = 0, timeout_seconds: float = 30.0
    ) -> dict[str, Any]:
        """Wait for a requested canonical revision to reach native projection."""

        return _call_tool(
            "projection_barrier",
            {
                "record_id": record_id,
                "commit_revision": commit_revision,
                "timeout_seconds": timeout_seconds,
            },
        )

    @server.tool()
    def read_working_set(
        owner_id: str,
        generation_id: str,
        record_ids: list[str],
        max_tokens: int = 2048,
        max_bytes: int = 32768,
    ) -> dict[str, Any]:
        """Read exact owned candidate IDs; their content remains unverified working state."""

        return _call_tool(
            "read_working_set",
            {
                "owner_id": owner_id,
                "generation_id": generation_id,
                "record_ids": record_ids,
                "max_tokens": max_tokens,
                "max_bytes": max_bytes,
            },
        )

    @server.tool()
    def build_context(
        query: str,
        limit: int = 4,
        tags: list[str] | None = None,
        source_horizon: SourceHorizon = "durable",
        include_candidates: bool = False,
        logical_block_ids: list[str] | None = None,
        search_all: bool = False,
        retrieval_mode: RetrievalMode = "semantic",
        record_ids: list[str] | None = None,
        max_tokens: int = 2048,
        source_speakers: list[str] | None = None,
        temporal_mode: str = "current",
        as_of: str = "",
        known_as_of: str = "",
        revision_id: str = "",
    ) -> dict[str, Any]:
        """Build small prompt-safe context from inspected leaves; expand only if insufficient."""

        return _call_tool(
            "build_context",
            {
                "query": query,
                "limit": limit,
                "tags": tags,
                "source_horizon": source_horizon,
                "include_candidates": include_candidates,
                "logical_block_ids": logical_block_ids,
                "search_all": search_all,
                "retrieval_mode": retrieval_mode,
                "record_ids": record_ids,
                "max_tokens": max_tokens,
                "source_speakers": source_speakers,
                "temporal_mode": temporal_mode,
                "as_of": as_of,
                "known_as_of": known_as_of,
                "revision_id": revision_id,
            },
        )

    @server.tool()
    def forget(
        memory_id: str,
        reason: str = "user_requested",
    ) -> dict[str, Any]:
        """Retract a memory block while preserving audit metadata."""

        return _call_tool(
            "forget",
            {
                "memory_id": memory_id,
                "reason": reason,
            },
        )

    @server.tool()
    def add_edge(
        src_id: str,
        dst_id: str,
        edge_type: str,
        reason: str = "",
        weight: float = 1.0,
        evidence_refs: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a typed Lethyx graph edge between two memory blocks."""

        return _call_tool(
            "add_edge",
            {
                "src_id": src_id,
                "dst_id": dst_id,
                "edge_type": edge_type,
                "reason": reason,
                "weight": weight,
                "evidence_refs": evidence_refs,
                "metadata": metadata,
            },
        )

    @server.tool()
    def export_graph() -> dict[str, Any]:
        """Export the scoped selected-backend graph without opening another transport."""

        return _call_tool("export_graph", {})

    @server.tool()
    def export_obsidian(
        vault_dir: str = "",
        project_title: str = "",
    ) -> dict[str, Any]:
        """Export scoped Lethyx memory to the canonical Obsidian live memory view."""

        return _call_tool(
            "export_obsidian",
            {
                "vault_dir": vault_dir,
                "project_title": project_title,
            },
        )

    _make_tool_arguments_strict(server)
    return server


mcp = create_mcp_server()


def main() -> None:
    """Run the Lethyx MCP server over stdio."""

    mcp.run()


if __name__ == "__main__":
    main()
