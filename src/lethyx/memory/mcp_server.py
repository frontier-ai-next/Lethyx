"""Lethyx MCP server.

Run with:

    PYTHONPATH=src python -m lethyx.memory.mcp_server

Hermes will expose these tools as `mcp_<server_name>_<tool_name>`.
"""

import json
import os
import subprocess
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

TOOL_NAMES = (
    "health",
    "remember",
    "recall",
    "build_context",
    "promote",
    "forget",
    "add_edge",
    "export_obsidian",
)


def _tool_timeout() -> int:
    try:
        return max(1, int(os.getenv("LETHYX_MCP_TOOL_TIMEOUT_SECONDS", "120")))
    except ValueError:
        return 120


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a Lethyx tool in a short-lived subprocess.

    This keeps the stdio MCP server responsive even when a native backend uses
    libraries that do not behave well inside the MCP worker thread on Windows.
    """

    process = subprocess.run(
        [sys.executable, "-m", "lethyx.memory.mcp_call", name],
        input=json.dumps(arguments, ensure_ascii=False),
        text=True,
        capture_output=True,
        timeout=_tool_timeout(),
        env=os.environ.copy(),
    )
    stdout = process.stdout.strip()
    if process.returncode != 0:
        return {
            "ok": False,
            "error": f"Lethyx MCP subprocess exited {process.returncode}",
            "stderr": process.stderr.strip()[-1000:],
            "stdout": stdout[-1000:],
        }
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error": f"Lethyx MCP subprocess returned invalid JSON: {exc}",
            "stdout": stdout[-1000:],
            "stderr": process.stderr.strip()[-1000:],
        }
    return (
        payload
        if isinstance(payload, dict)
        else {"ok": False, "error": "tool returned non-object JSON"}
    )


def create_mcp_server() -> FastMCP:
    """Create the stdio MCP server with project-scoped Lethyx tools."""

    server = FastMCP("lethyx")

    @server.tool()
    def health(workspace_id: str = "", backend: str = "", repo_root: str = "") -> dict[str, Any]:
        """Check the selected Lethyx backend for a workspace."""

        return _call_tool(
            "health", {"workspace_id": workspace_id, "backend": backend, "repo_root": repo_root}
        )

    @server.tool()
    def remember(
        content: str,
        title: str = "",
        workspace_id: str = "",
        backend: str = "",
        repo_root: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        evidence_refs: list[Any] | None = None,
        scope: str = "workspace",
        lifecycle: str = "active",
        surface: str = "supplemental_evidence",
        trust_level: str = "agent_claim",
        memory_id: str = "",
        run_id: str = "",
        phase_id: str = "",
        subtask_id: str = "",
    ) -> dict[str, Any]:
        """Store one scoped Lethyx memory block as non-directive evidence by default."""

        return _call_tool(
            "remember",
            {
                "content": content,
                "title": title,
                "workspace_id": workspace_id,
                "backend": backend,
                "repo_root": repo_root,
                "tags": tags,
                "metadata": metadata,
                "evidence_refs": evidence_refs,
                "scope": scope,
                "lifecycle": lifecycle,
                "surface": surface,
                "trust_level": trust_level,
                "memory_id": memory_id,
                "run_id": run_id,
                "phase_id": phase_id,
                "subtask_id": subtask_id,
            },
        )

    @server.tool()
    def recall(
        query: str,
        workspace_id: str = "",
        backend: str = "",
        repo_root: str = "",
        limit: int = 10,
        tags: list[str] | None = None,
        include_global: bool = False,
        include_passive: bool = False,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Recall project-scoped memory. Global memory is excluded unless requested."""

        return _call_tool(
            "recall",
            {
                "query": query,
                "workspace_id": workspace_id,
                "backend": backend,
                "repo_root": repo_root,
                "limit": limit,
                "tags": tags,
                "include_global": include_global,
                "include_passive": include_passive,
                "max_tokens": max_tokens,
            },
        )

    @server.tool()
    def build_context(
        query: str,
        workspace_id: str = "",
        backend: str = "",
        repo_root: str = "",
        limit: int = 8,
        include_global: bool = False,
        include_passive: bool = False,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Build a prompt-safe, non-directive context bundle from scoped memory."""

        return _call_tool(
            "build_context",
            {
                "query": query,
                "workspace_id": workspace_id,
                "backend": backend,
                "repo_root": repo_root,
                "limit": limit,
                "include_global": include_global,
                "include_passive": include_passive,
                "max_tokens": max_tokens,
            },
        )

    @server.tool()
    def promote(
        memory_id: str,
        reason: str,
        evidence_refs: list[Any],
        target_surface: str = "supplemental_evidence",
        workspace_id: str = "",
        backend: str = "",
        repo_root: str = "",
    ) -> dict[str, Any]:
        """Promote a memory block with explicit evidence."""

        return _call_tool(
            "promote",
            {
                "memory_id": memory_id,
                "reason": reason,
                "evidence_refs": evidence_refs,
                "target_surface": target_surface,
                "workspace_id": workspace_id,
                "backend": backend,
                "repo_root": repo_root,
            },
        )

    @server.tool()
    def forget(
        memory_id: str,
        reason: str = "user_requested",
        workspace_id: str = "",
        backend: str = "",
        repo_root: str = "",
    ) -> dict[str, Any]:
        """Retract a memory block while preserving audit metadata."""

        return _call_tool(
            "forget",
            {
                "memory_id": memory_id,
                "reason": reason,
                "workspace_id": workspace_id,
                "backend": backend,
                "repo_root": repo_root,
            },
        )

    @server.tool()
    def add_edge(
        src_id: str,
        dst_id: str,
        edge_type: str,
        workspace_id: str = "",
        backend: str = "",
        repo_root: str = "",
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
                "workspace_id": workspace_id,
                "backend": backend,
                "repo_root": repo_root,
                "reason": reason,
                "weight": weight,
                "evidence_refs": evidence_refs,
                "metadata": metadata,
            },
        )

    @server.tool()
    def export_obsidian(
        workspace_id: str = "",
        backend: str = "",
        repo_root: str = "",
        vault_dir: str = "",
        project_title: str = "",
        include_global: bool = False,
    ) -> dict[str, Any]:
        """Export scoped Lethyx memory to the canonical Obsidian live memory view."""

        return _call_tool(
            "export_obsidian",
            {
                "workspace_id": workspace_id,
                "backend": backend,
                "repo_root": repo_root,
                "vault_dir": vault_dir,
                "project_title": project_title,
                "include_global": include_global,
            },
        )

    return server


mcp = create_mcp_server()


def main() -> None:
    """Run the Lethyx MCP server over stdio."""

    mcp.run()


if __name__ == "__main__":
    main()
