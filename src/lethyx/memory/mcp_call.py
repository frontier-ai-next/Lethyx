"""Subprocess-safe dispatcher for Lethyx MCP tools.

Some native memory backends (notably Chroma-backed MemPalace on Windows) can block
when initialized inside an MCP server worker thread.  The stdio MCP server uses
this tiny runner to execute backend work in a short-lived Python process while
keeping the MCP process itself responsive.
"""

import json
import sys
from collections.abc import Callable
from typing import Any

from lethyx.memory.mcp_facade import (
    lethyx_add_edge,
    lethyx_build_context,
    lethyx_export_obsidian,
    lethyx_forget,
    lethyx_health,
    lethyx_promote,
    lethyx_recall,
    lethyx_remember,
)

TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "health": lethyx_health,
    "remember": lethyx_remember,
    "recall": lethyx_recall,
    "build_context": lethyx_build_context,
    "promote": lethyx_promote,
    "forget": lethyx_forget,
    "add_edge": lethyx_add_edge,
    "export_obsidian": lethyx_export_obsidian,
}


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = TOOLS.get(name)
    if tool is None:
        return {"ok": False, "error": f"unknown Lethyx MCP tool: {name}"}
    try:
        return tool(**arguments)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print(json.dumps({"ok": False, "error": "tool name is required"}), flush=True)
        return 2
    raw = sys.stdin.read().strip() or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": f"invalid JSON arguments: {exc}"}), flush=True)
        return 2
    if not isinstance(payload, dict):
        print(json.dumps({"ok": False, "error": "arguments must be a JSON object"}), flush=True)
        return 2
    print(json.dumps(call_tool(args[0], payload), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
