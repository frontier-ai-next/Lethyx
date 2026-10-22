"""Local operator panel on the existing authenticated memory listener.

The browser is a view, not a second memory store. Writes traverse the same
leased dispatcher as MCP. A bounded, content-free event ring invalidates views;
reconnection outside the ring explicitly requests a fresh canonical snapshot.
"""

import argparse
import hmac
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime
from http.cookies import SimpleCookie
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from um_mem.memory.local_service import (
    LOCAL_SERVICE_API_VERSION,
    LOCAL_SERVICE_SCHEMA_VERSION,
    LocalServiceClientPool,
    _local_service_http_request,
    build_local_service_identity,
    local_service_bootstrap_secret,
    local_service_registry,
)

_ASSETS = {"app.js": "text/javascript", "style.css": "text/css"}
_READS = {"recall", "build_context", "build_granted_context", "resolve_active_mask"}
_QUIET = {
    "health",
    "block_map",
    "expand_block",
    "navigation_envelope",
    "recent",
    "list_records",
    "read_exact_record",
    "read_exact_records",
    "get_active_mask",
    "operation_status",
    "projection_barrier",
    "memory_map",
    "procedure",
    "read_working_set",
    "export_graph",
    "export_obsidian",
}


class ChangeFeed:
    """Monotonic cursor within one listener lifetime; no record text is retained."""

    def __init__(self, capacity: int = 256) -> None:
        self.epoch = secrets.token_hex(8)
        self.sequence = 0
        self.events: deque[dict[str, Any]] = deque(maxlen=capacity)
        self.condition = threading.Condition()

    @property
    def cursor(self) -> str:
        with self.condition:
            return f"{self.epoch}:{self.sequence}"

    def publish(self, event: dict[str, Any]) -> None:
        with self.condition:
            self.sequence += 1
            self.events.append({**event, "id": self.cursor, "at": datetime.now(UTC).isoformat()})
            self.condition.notify_all()

    def since(self, cursor: str) -> tuple[bool, list[dict[str, Any]]]:
        with self.condition:
            try:
                epoch, number = cursor.split(":")
                sequence = int(number)
            except (ValueError, AttributeError):
                return True, []
            oldest = self.sequence - len(self.events)
            if epoch != self.epoch or not oldest <= sequence <= self.sequence:
                return True, []
            return False, [e for e in self.events if int(e["id"].split(":")[1]) > sequence]


def _summary(record: Any) -> dict[str, Any]:
    membership = record.block_membership
    return {
        "record_id": record.block_id,
        "revision_id": record.revision_id,
        "revision_of": record.revision_of,
        "title": record.title,
        "preview": record.content[:240],
        "kind": record.kind,
        "status": record.status,
        "recorded_at": record.recorded_at,
        "logical_block_id": membership.primary_block_id if membership else "",
        "tags": list(record.tags),
        "authority": record.authority,
    }


class Dashboard:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.feed = ChangeFeed()
        self.access_token = secrets.token_urlsafe(32)
        self.sessions: dict[str, float] = {}
        self.session_lock = threading.Lock()
        self.stream_slots = threading.BoundedSemaphore(16)
        self.pool = LocalServiceClientPool()
        self.active = False
        self.last_browser_at = 0.0
        self.stream_count = 0

    def activate(self) -> None:
        with self.session_lock:
            if not self.active:
                self.sessions.clear()
                self.access_token = secrets.token_urlsafe(32)
                self.feed = ChangeFeed()
                self.active = True
            self.last_browser_at = time.monotonic()

    def deactivate(self, *, idle_only: bool = False) -> None:
        with self.session_lock:
            if idle_only and (
                not self.active
                or self.stream_count
                or time.monotonic() - self.last_browser_at <= 60
            ):
                return
            self.active = False
            self.sessions.clear()
            self.access_token = secrets.token_urlsafe(32)
        with self.feed.condition:
            self.feed.condition.notify_all()
        self.pool.close()

    def expire_if_idle(self) -> None:
        self.deactivate(idle_only=True)

    def observe(self, tool: str, arguments: dict, result: dict, lineage: Any) -> None:
        if not self.active or tool in _QUIET:
            return
        items = result.get("hits") or result.get("items") or []
        record_ids = []
        for item in items[:32]:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") or {}
            record_id = item.get("record_id") or metadata.get("record_id") or item.get("source")
            if record_id:
                record_ids.append(str(record_id))
        self.feed.publish(
            {
                "kind": "retrieval" if tool in _READS else "mutation",
                "operation": tool,
                "ok": result.get("ok") is True,
                "session_id": lineage.session_id,
                "task_id": lineage.task_id,
                "record_id": str(
                    result.get("memory_id")
                    or result.get("block_id")
                    or arguments.get("memory_id")
                    or ""
                ),
                "record_ids": record_ids,
                "count": len(items),
                "rendered": bool(result.get("rendered_context")),
                "commit_revision": result.get("commit_revision"),
                "projection_revision": result.get("projection_revision"),
                "reason_code": result.get("reason_code", ""),
            }
        )

    def snapshot(self) -> dict[str, Any]:
        # Capture the cursor first: a concurrent commit will be replayed after
        # this read, even if the read happened to observe the new state already.
        cursor = self.feed.cursor
        client = self.runtime.client
        catalog = client.catalog_snapshot()
        nodes = {b["block_id"]: {**b, "children": []} for b in catalog["blocks"]}
        tree: list[dict[str, Any]] = []
        for node in nodes.values():
            parent = nodes.get(node.get("parent_id"))
            (parent["children"] if parent else tree).append(node)
        records = client.recent_records(limit=64, include_candidates=True)
        with self.feed.condition:
            events = list(self.feed.events)
        with self.runtime._leases_lock:
            self.runtime._expire_leases()
            sessions = {
                lease.session_id: asdict(lease.scope_lineage)
                for lease in self.runtime._leases.values()
            }
        return {
            "ok": True,
            "cursor": cursor,
            "tree": tree,
            "edges": catalog["edges"],
            "records": [_summary(r) for r in records],
            "recent_limit": 64,
            "workspace": client.workspace_id,
            "profile": client.profile_id,
            "backend": self.runtime.identity.backend,
            "health": self.runtime.health(),
            "events": events,
            "sessions": list(sessions.values()),
        }

    def records(self, block: str, cursor: str) -> dict[str, Any]:
        client = self.runtime.client
        if not block:
            raise ValueError("Select a topic to page through its records")
        page = client.list_records(
            scope="workspace",
            scope_id=client.workspace_id,
            logical_block_id=block,
            limit=64,
            cursor=cursor,
        )
        ids = [r["record_id"] for r in page["records"]]
        records = client.read_exact_records(ids) if ids else []
        return {**page, "ok": True, "records": [_summary(r) for r in records]}

    def record(self, record_id: str) -> dict[str, Any]:
        client = self.runtime.client
        record = client.read_exact_record(record_id=record_id)
        history_reader = getattr(client.backend, "canonical_history", None)
        history = (
            history_reader(canonical_key=record.canonical_key, workspace_id=client.workspace_id)
            if callable(history_reader) and record.canonical_key
            else [record]
        )
        history = [r for r in history if r.profile_id == client.profile_id]
        current = client._current_record_id_set(history) or set()
        return {
            "ok": True,
            "record": asdict(record),
            "is_current": record.block_id in current,
            "history": [
                {**_summary(r), "is_current": r.block_id in current} for r in history[-128:]
            ],
            "history_truncated": len(history) > 128,
        }

    def call(self, session: str, body: dict[str, Any]) -> dict[str, Any]:
        tool = body.get("tool")
        arguments = body.get("arguments")
        if tool not in {"recall", "amend", "remember", "correct"} or not isinstance(
            arguments, dict
        ):
            raise ValueError("Unsupported panel operation")
        allowed = {
            "recall": {"query", "logical_block_ids", "include_candidates"},
            "amend": {
                "memory_id",
                "expected_revision",
                "reason",
                "content",
                "title",
                "operation_id",
            },
            "remember": {"content", "title", "requested_block_id", "operation_id"},
        }["amend" if tool == "correct" else tool]
        if set(arguments) - allowed:
            raise ValueError("Unsupported panel arguments")
        arguments = dict(arguments)
        if tool == "recall":
            arguments.update(limit=20, search_all=not bool(arguments.get("logical_block_ids")))
        else:
            for key in ("title", "content", "operation_id"):
                if not isinstance(arguments.get(key), str) or not arguments[key].strip():
                    raise ValueError(f"{key} is required")
            if len(arguments["content"]) > 100_000 or len(arguments["title"]) > 500:
                raise ValueError("Record exceeds panel editing limit")
            if tool in {"amend", "correct"} and not all(
                arguments.get(k) for k in ("memory_id", "expected_revision", "reason")
            ):
                raise ValueError("Amendment requires an exact version and reason")
            if tool == "remember":
                if not arguments.get("requested_block_id"):
                    raise ValueError("Choose a writable topic")
                arguments.update(kind="observation", source_id=f"panel:{session[:16]}")
            if tool == "correct":
                record = self.runtime.client.read_exact_record(record_id=arguments["memory_id"])
                arguments.update(
                    operation="correct", requested_block_id=record.block_membership.primary_block_id
                )
                tool = "remember"
        result = self.pool.call(
            self.runtime.identity,
            tool,
            arguments,
            session_id=f"panel:{session[:24]}",
            mutation_correlation_id=str(arguments.get("operation_id") or ""),
            timeout_seconds=120,
        )
        # Older public amend implementations report the initial CAS check as
        # ValueError rather than the typed conflict raised by the writer lane.
        if (
            tool == "amend"
            and result.get("ok") is False
            and "amend expected_revision does not match" in str(result.get("error") or "")
        ):
            result["reason_code"] = "canonical_revision_conflict"
        if result.get("ok") and result.get("canonical_reference"):
            result.setdefault("memory_id", result["canonical_reference"])
        return result

    def _session(self, handler: Any) -> str:
        cookie = SimpleCookie()
        try:
            cookie.load(handler.headers.get("Cookie", ""))
            name = f"um_mem_panel_{handler.server.server_port}"
            token = cookie[name].value if name in cookie else ""
        except (ValueError, TypeError):
            return ""
        with self.session_lock:
            expiry = self.sessions.get(token, 0)
            if expiry < time.monotonic():
                self.sessions.pop(token, None)
                return ""
        return token

    def handle(self, handler: Any) -> None:
        """Serve only the fixed panel namespace, with no filesystem paths from HTTP."""
        origin = f"http://127.0.0.1:{handler.server.server_port}"
        if (
            handler.headers.get("Host") != origin.removeprefix("http://")
            or handler.headers.get("Origin", origin) != origin
            or handler.headers.get("Sec-Fetch-Site", "same-origin") == "cross-site"
        ):
            self._send(handler, {"error": "Same-origin loopback access required"}, 403)
            return
        route = urlsplit(handler.path)
        try:
            self.expire_if_idle()
            if route.path != "/panel/connect" and not self.active:
                self._send(handler, {"error": "Panel is stopped. Run um-memory dashboard."}, 410)
                return
            if handler.command == "GET" and route.path in {"/panel", "/panel/"}:
                self._asset(handler, "index.html", "text/html")
                return
            if handler.command == "GET" and route.path.startswith("/panel/assets/"):
                name = route.path.removeprefix("/panel/assets/")
                if name in _ASSETS:
                    self._asset(handler, name, _ASSETS[name])
                else:
                    self._send(handler, {"error": "Unknown asset"}, 404)
                return
            body: dict[str, Any] = {}
            if handler.command == "POST":
                if handler.headers.get_content_type() != "application/json":
                    self._send(handler, {"error": "JSON required"}, 415)
                    return
                size = int(handler.headers.get("Content-Length", "0"))
                if not 0 < size <= 524288:
                    self._send(handler, {"error": "Invalid request size"}, 413)
                    return
                body = json.loads(handler.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
            if handler.command == "POST" and route.path == "/panel/connect":
                secret = body.get("bootstrap_secret")
                if (
                    self.runtime._frame_error(body) is not None
                    or body.get("caller_role") != "operator"
                    or not isinstance(secret, str)
                    or not hmac.compare_digest(
                        secret, local_service_bootstrap_secret(self.runtime.identity)
                    )
                ):
                    self._send(handler, {"error": "Operator authentication required"}, 403)
                    return
                if body.get("action") == "stop":
                    self.deactivate()
                    self._send(handler, {"ok": True, "state": "stopped"})
                else:
                    self.activate()
                    self._send(
                        handler, {"ok": True, "url": f"{origin}/panel/#" + self.access_token}
                    )
                return
            if handler.command == "POST" and route.path == "/panel/auth":
                token = body.get("token")
                session = secrets.token_urlsafe(32)
                with self.session_lock:
                    if (
                        not self.active
                        or not isinstance(token, str)
                        or not hmac.compare_digest(token, self.access_token)
                    ):
                        self._send(
                            handler, {"error": "Open the panel with um-memory dashboard"}, 401
                        )
                        return
                    self.sessions = {k: v for k, v in self.sessions.items() if v > time.monotonic()}
                    if len(self.sessions) >= 128:
                        self._send(handler, {"error": "Too many panel sessions"}, 429)
                        return
                    self.sessions[session] = time.monotonic() + 43200
                self._send(
                    handler,
                    {"ok": True},
                    cookie=(
                        f"um_mem_panel_{handler.server.server_port}={session}; Path=/panel; HttpOnly; SameSite=Strict; Max-Age=43200"
                    ),
                )
                return
            session = self._session(handler)
            if not session:
                self._send(handler, {"error": "Open the panel with um-memory dashboard"}, 401)
                return
            self.runtime._touch_work()
            self.last_browser_at = time.monotonic()
            query = parse_qs(route.query)

            def value(key: str) -> str:
                return query.get(key, [""])[0]

            if handler.command == "POST" and route.path == "/panel/api/stop":
                self.deactivate()
                self._send(handler, {"ok": True, "state": "stopped"})
            elif handler.command == "GET" and route.path == "/panel/api/events":
                self._events(handler, handler.headers.get("Last-Event-ID") or value("since"))
            elif handler.command == "GET" and route.path == "/panel/api/snapshot":
                self._send(handler, self.snapshot())
            elif handler.command == "GET" and route.path == "/panel/api/records":
                self._send(handler, self.records(value("block"), value("cursor")))
            elif handler.command == "GET" and route.path == "/panel/api/record":
                self._send(handler, self.record(value("id")))
            elif handler.command == "POST" and route.path == "/panel/api/call":
                result = self.call(session, body)
                code = 409 if result.get("reason_code") == "canonical_revision_conflict" else 200
                self._send(handler, result, code)
            else:
                self._send(handler, {"error": "Unknown panel endpoint"}, 404)
        except (ValueError, TypeError, KeyError) as exc:
            self._send(handler, {"ok": False, "error": str(exc)}, 400)
        except (OSError, RuntimeError):
            self._send(handler, {"ok": False, "error": "Memory service unavailable; retry"}, 503)

    def _events(self, handler: Any, cursor: str) -> None:
        if not self.stream_slots.acquire(blocking=False):
            self._send(handler, {"error": "Too many live panels"}, 429)
            return
        with self.session_lock:
            self.stream_count += 1
        feed = self.feed
        try:
            handler.send_response(200)
            self._headers(handler, "text/event-stream")
            handler.end_headers()
            while self.active and self.feed is feed and not self.runtime._draining:
                reset, events = feed.since(cursor)
                if reset:
                    cursor = feed.cursor
                    handler.wfile.write(f"id: {cursor}\nevent: reset\ndata: {{}}\n\n".encode())
                for event in events:
                    cursor = event["id"]
                    payload = json.dumps(event, ensure_ascii=False)
                    handler.wfile.write(
                        f"id: {cursor}\nevent: change\ndata: {payload}\n\n".encode()
                    )
                handler.wfile.write(b": heartbeat\n\n")
                handler.wfile.flush()
                self.runtime._touch_work()
                with feed.condition:
                    # Recheck under the lock so a commit between read and wait
                    # cannot be missed until the next heartbeat.
                    if cursor == feed.cursor and self.active:
                        feed.condition.wait(timeout=15)
                if not self._session(handler):
                    if not self.active:
                        handler.wfile.write(b"event: stopped\ndata: {}\n\n")
                        handler.wfile.flush()
                    return
            handler.wfile.write(b"event: stopped\ndata: {}\n\n")
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionError, OSError):
            return
        finally:
            self.stream_slots.release()
            with self.session_lock:
                self.stream_count -= 1
                self.last_browser_at = time.monotonic()

    @staticmethod
    def _headers(handler: Any, content_type: str) -> None:
        handler.send_header("Content-Type", content_type + "; charset=utf-8")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.send_header("Referrer-Policy", "no-referrer")
        handler.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; "
            "style-src 'self'; connect-src 'self'; img-src 'self' data:; "
            "object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )

    def _asset(self, handler: Any, name: str, content_type: str) -> None:
        payload = files("um_mem.memory").joinpath("panel", name).read_bytes()
        handler.send_response(200)
        self._headers(handler, content_type)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _send(self, handler: Any, payload: dict, status: int = 200, cookie: str = "") -> None:
        raw = json.dumps(payload, ensure_ascii=False, default=str).encode()
        handler.send_response(status)
        self._headers(handler, "application/json")
        if cookie:
            handler.send_header("Set-Cookie", cookie)
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)


def dashboard_url(
    repo_root: str, workspace: str, profile: str, backend: str, *, stop: bool = False
) -> dict:
    """Open a browser capability via local operator authentication, never a URL query."""
    identity = build_local_service_identity(
        repo_root=repo_root,
        workspace_id=workspace,
        profile_id=profile,
        backend=backend,
        caller_role="operator",
    )
    if (
        not stop
        and local_service_registry(identity) is None
        and not os.environ.get("UM_MEM_LOCAL_SERVICE_OWNER_PID")
    ):
        _start_panel_owner(identity)
    pool = LocalServiceClientPool()
    try:
        if not stop:
            health = pool.call(
                identity, "health", {}, session_id="panel-launcher", timeout_seconds=120
            )
            if not health.get("ok"):
                return health
        endpoint = local_service_registry(identity)
        if endpoint is None:
            if stop:
                return {"ok": True, "state": "stopped"}
            return {"ok": False, "error": "Local service is unavailable"}
        return _local_service_http_request(
            endpoint,
            "/panel/connect",
            {
                "protocol_version": identity.protocol_version,
                "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
                "api_version": LOCAL_SERVICE_API_VERSION,
                "request_id": "panel:" + secrets.token_hex(16),
                "caller_role": "operator",
                "bootstrap_secret": local_service_bootstrap_secret(identity),
                "action": "stop" if stop else "start",
            },
            timeout_seconds=10,
        )
    finally:
        pool.close()


def _start_panel_owner(identity: Any) -> None:
    """Keep a command-launched listener alive after the CLI exits."""
    executable = sys.executable
    if os.name == "nt":
        pythonw = str(Path(executable).with_name("pythonw.exe"))
        if Path(pythonw).is_file():
            executable = pythonw
    process = subprocess.Popen(
        [
            executable,
            "-m",
            "um_mem.memory.dashboard",
            "--serve-only",
            "--repo-root",
            str(identity.repo_root),
            "--workspace",
            identity.workspace_id,
            "--profile",
            identity.profile_id,
            "--backend",
            identity.backend,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        close_fds=True,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if local_service_registry(identity) is not None:
            return
        if process.poll() is not None:
            raise RuntimeError("UM-MEM panel owner exited before the listener started")
        time.sleep(0.1)
    raise TimeoutError("UM-MEM panel owner did not start within 20 seconds")


def main(argv: list[str] | None = None) -> int:
    """Windowless-friendly panel launcher; own a new listener until it goes idle."""
    import webbrowser

    from um_mem.memory.settings import default_store, load_config, mcp_environment

    parser = argparse.ArgumentParser(prog="python -m um_mem.memory.dashboard")
    parser.add_argument("--serve-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--workspace", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--backend", default="")
    args = parser.parse_args(argv)
    config = load_config()
    for key, value in mcp_environment(config).items():
        os.environ.setdefault(key, value)
    os.environ["UM_MEM_LOCAL_SERVICE_OWNER_PID"] = str(os.getpid())
    repo = str(args.repo_root or config.get("repo_root") or default_store())
    workspace = str(args.workspace or config.get("workspace") or "default")
    profile = str(args.profile or config.get("profile") or "default")
    backend = str(args.backend or config.get("backend") or "mnemosyne")
    identity = build_local_service_identity(
        repo_root=repo,
        workspace_id=workspace,
        profile_id=profile,
        backend=backend,
        caller_role="operator",
    )
    if args.serve_only:
        pool = LocalServiceClientPool()
        try:
            result = pool.call(
                identity, "health", {}, session_id="panel-owner", timeout_seconds=120
            )
        finally:
            pool.close()
    else:
        result = dashboard_url(repo, workspace, profile, backend)
        if result.get("ok"):
            webbrowser.open(str(result["url"]))
    if not result.get("ok"):
        print(json.dumps(result))
        return 1
    while True:
        endpoint = local_service_registry(identity)
        if endpoint is None or endpoint.get("owner_pid") != os.getpid():
            return 0
        # Do not keep the service alive artificially. Browser SSE heartbeats
        # touch it; once all clients close, the existing idle reaper stops it.
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
