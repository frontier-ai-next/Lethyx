"""Live panel protocol: replay, host authentication, isolation and governed edits."""

import http.client
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from um_mem.memory import dashboard, mcp_call
from um_mem.memory.scopes import ScopeLineage


def test_change_feed_replays_and_resets_across_gaps_and_restart():
    feed = dashboard.ChangeFeed(capacity=2)
    initial = feed.cursor
    feed.publish({"operation": "remember"})
    first = feed.cursor
    feed.publish({"operation": "amend"})
    assert [e["operation"] for e in feed.since(initial)[1]] == ["remember", "amend"]
    feed.publish({"operation": "forget"})
    assert feed.since(initial) == (True, [])
    assert [e["operation"] for e in feed.since(first)[1]] == ["amend", "forget"]
    assert dashboard.ChangeFeed().since(feed.cursor) == (True, [])
    assert feed.since(f"{feed.epoch}:999") == (True, [])


def test_disconnecting_before_stream_headers_does_not_exhaust_live_panels():
    panel = dashboard.Dashboard(SimpleNamespace(_draining=False))
    responses = []

    def disconnect():
        raise BrokenPipeError("Browser closed before headers were sent")

    handler = SimpleNamespace(
        send_response=responses.append,
        send_header=lambda *args: None,
        end_headers=disconnect,
    )
    for _ in range(32):
        panel._events(handler, panel.feed.cursor)
    assert responses == [200] * 32


def test_events_never_copy_queries_or_record_content_and_reads_do_not_invalidate():
    panel = dashboard.Dashboard(SimpleNamespace())
    panel.activate()
    lineage = ScopeLineage(workspace_id="ws", session_id="codex:one", task_id="task")
    panel.observe(
        "recall",
        {"query": "private query"},
        {
            "ok": True,
            "hits": [{"source": "mem-1", "text": "private content"}],
        },
        lineage,
    )
    panel.observe("read_exact_record", {}, {"ok": True, "record": "private"}, lineage)
    events = list(panel.feed.events)
    assert len(events) == 1
    assert events[0]["record_ids"] == ["mem-1"]
    assert events[0]["session_id"] == "codex:one"
    assert "private" not in json.dumps(events)


@pytest.fixture
def panel_server(monkeypatch):
    identity = SimpleNamespace(backend="mnemosyne")
    runtime = SimpleNamespace(
        identity=identity,
        client=SimpleNamespace(),
        _draining=False,
        _touch_work=lambda: None,
        _frame_error=lambda body: None,
    )
    runtime.dashboard = dashboard.Dashboard(runtime)
    runtime.dashboard.activate()
    monkeypatch.setattr(dashboard, "local_service_bootstrap_secret", lambda _: "operator-secret")
    server = mcp_call._LocalServiceListenerServer(
        ("127.0.0.1", 0), mcp_call._listener_handler(runtime)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"

    def request(path, body=None, headers=None, method=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        selected = {"Content-Type": "application/json", **(headers or {})}
        connection.request(
            method or ("POST" if body is not None else "GET"),
            path,
            json.dumps(body) if body is not None else None,
            selected,
        )
        response = connection.getresponse()
        result = (response.status, dict(response.getheaders()), response.read())
        connection.close()
        return result

    yield runtime.dashboard, request, origin, server
    runtime._draining = True
    with runtime.dashboard.feed.condition:
        runtime.dashboard.feed.condition.notify_all()
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def authenticate(panel, request):
    status, headers, _ = request("/panel/auth", {"token": panel.access_token})
    assert status == 200
    assert "HttpOnly" in headers["Set-Cookie"]
    assert "SameSite=Strict" in headers["Set-Cookie"]
    return {"Cookie": headers["Set-Cookie"].split(";", 1)[0]}


def test_panel_is_on_demand_and_stopping_does_not_drain_memory(panel_server):
    panel, request, _, _ = panel_server
    panel.deactivate()
    assert request("/panel/")[0] == 410
    assert request("/panel/assets/app.js")[0] == 410
    assert not panel.runtime._draining
    status, _, body = request(
        "/panel/connect", {"caller_role": "operator", "bootstrap_secret": "operator-secret"}
    )
    assert status == 200 and json.loads(body)["url"]
    cookie = authenticate(panel, request)
    assert request("/panel/")[0] == 200
    assert request("/panel/api/stop", {}, headers=cookie)[0] == 200
    assert request("/panel/api/snapshot", headers=cookie)[0] == 410
    assert not panel.runtime._draining


def test_inactive_panel_collects_no_events_and_expires_without_browsers(monkeypatch):
    panel = dashboard.Dashboard(SimpleNamespace())
    lineage = ScopeLineage(workspace_id="ws", session_id="test")
    panel.observe("remember", {}, {"ok": True}, lineage)
    assert not panel.active and not panel.feed.events
    panel.activate()
    panel.stream_count = 1
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: panel.last_browser_at + 61)
    panel.expire_if_idle()
    assert panel.active
    panel.stream_count = 0
    panel.expire_if_idle()
    assert not panel.active


def test_auth_cannot_create_a_session_after_concurrent_stop(panel_server, monkeypatch):
    panel, request, _, _ = panel_server
    old_token = panel.access_token
    original = dashboard.secrets.token_urlsafe
    interrupted = False

    def stop_during_auth(size):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            panel.deactivate()
        return original(size)

    monkeypatch.setattr(dashboard.secrets, "token_urlsafe", stop_during_auth)
    assert request("/panel/auth", {"token": old_token})[0] == 401
    assert not panel.sessions
    panel.activate()
    assert request("/panel/auth", {"token": old_token})[0] == 401


def test_panel_requires_operator_bootstrap_and_rejects_foreign_origins(panel_server):
    panel, request, origin, _ = panel_server
    assert request("/panel/api/snapshot")[0] == 401
    assert (
        request("/panel/connect", {"caller_role": "operator", "bootstrap_secret": "wrong"})[0]
        == 403
    )
    status, _, body = request(
        "/panel/connect",
        {
            "caller_role": "operator",
            "bootstrap_secret": "operator-secret",
        },
    )
    assert status == 200
    assert json.loads(body)["url"].startswith(origin + "/panel/#")
    headers = authenticate(panel, request)
    assert (
        request("/panel/api/snapshot", headers={**headers, "Origin": "https://attacker.test"})[0]
        == 403
    )
    assert request("/panel/api/snapshot", headers={**headers, "Host": "attacker.test"})[0] == 403
    assert (
        request("/panel/api/snapshot", headers={**headers, "Sec-Fetch-Site": "cross-site"})[0]
        == 403
    )
    assert request("/panel/assets/../../dashboard.py")[0] == 404
    status, headers, body = request("/panel/")
    assert status == 200
    assert b"operator-secret" not in body and panel.access_token.encode() not in body
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def test_panel_edits_keep_cas_and_do_not_accept_scope_or_authority_claims(
    panel_server, monkeypatch
):
    panel, request, _, _ = panel_server
    headers = authenticate(panel, request)
    calls = []

    def call(identity, tool, arguments, **kwargs):
        calls.append((tool, arguments, kwargs))
        return {"ok": False, "reason_code": "canonical_revision_conflict", "error": "stale"}

    monkeypatch.setattr(panel.pool, "call", call)
    arguments = {
        "memory_id": "mem-1",
        "expected_revision": "rev-old",
        "reason": "fix",
        "title": "title",
        "content": "content",
        "operation_id": "panel:123",
    }
    assert (
        request(
            "/panel/api/call",
            {
                "tool": "amend",
                "arguments": {
                    **arguments,
                    "workspace_id": "other",
                },
            },
            headers,
        )[0]
        == 400
    )
    assert request("/panel/api/call", {"tool": "promote", "arguments": {}}, headers)[0] == 400
    status, _, body = request("/panel/api/call", {"tool": "amend", "arguments": arguments}, headers)
    assert status == 409 and json.loads(body)["reason_code"] == "canonical_revision_conflict"
    assert calls[0][1]["expected_revision"] == "rev-old"
    assert calls[0][2]["mutation_correlation_id"] == "panel:123"
    assert calls[0][2]["session_id"].startswith("panel:")
    other_headers = authenticate(panel, request)
    request("/panel/api/call", {"tool": "amend", "arguments": arguments}, other_headers)
    assert calls[0][2]["session_id"] != calls[1][2]["session_id"]


def test_sse_replays_after_cursor_without_html_reload(panel_server):
    panel, request, _, server = panel_server
    headers = authenticate(panel, request)
    cursor = panel.feed.cursor
    panel.feed.publish({"kind": "mutation", "operation": "amend", "record_id": "mem-1"})
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request("GET", "/panel/api/events", headers={**headers, "Last-Event-ID": cursor})
    response = connection.getresponse()
    assert response.status == 200
    assert response.getheader("Content-Type").startswith("text/event-stream")
    assert response.fp.readline().startswith(b"id: ")
    assert response.fp.readline() == b"event: change\n"
    data = response.fp.readline()
    assert json.loads(data.removeprefix(b"data: "))["record_id"] == "mem-1"
    connection.close()


def test_snapshot_cursor_precedes_concurrent_commit():
    @dataclass
    class Identity:
        session_id: str = "codex"
        task_id: str = ""

    client = SimpleNamespace(workspace_id="ws", profile_id="default", recent_records=lambda **_: [])
    runtime = SimpleNamespace(
        client=client,
        identity=SimpleNamespace(backend="mnemosyne"),
        _leases_lock=threading.Lock(),
        _leases={},
        _expire_leases=lambda: None,
        health=lambda: {"ok": True},
    )
    panel = dashboard.Dashboard(runtime)
    panel.activate()

    def read_tree(**kwargs):
        panel.observe("amend", {}, {"ok": True}, Identity())
        return {"blocks": [], "edges": []}

    client.catalog_snapshot = read_tree
    snapshot = panel.snapshot()
    assert panel.feed.since(snapshot["cursor"])[1][0]["operation"] == "amend"


def _panel_record(record_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        block_id=record_id,
        revision_id=f"revision:{record_id}",
        revision_of=None,
        title=f"Record {record_id}",
        content="x" * 240,
        kind="fact",
        status="accepted",
        recorded_at="2026-10-23T00:00:00+00:00",
        block_membership=SimpleNamespace(primary_block_id="topic-1"),
        tags=(),
        authority="advisory",
    )


def _panel_for_records(records, edges_by_record, catalog=None):
    def edges_between(record_ids, *, limit):
        visible = set(record_ids)
        unique = {}
        for record_id in sorted(visible):
            for edge in edges_by_record.get(record_id, ()):
                if edge.src_id in visible and edge.dst_id in visible:
                    unique[(edge.src_id, edge.dst_id, edge.edge_type)] = edge
        return list(unique.values())[:limit]

    client = SimpleNamespace(
        workspace_id="ws",
        profile_id="profile-a",
        backend=SimpleNamespace(edges_between=edges_between),
        catalog_snapshot=lambda **_: (
            catalog or {"blocks": [], "edges": [], "edges_truncated": False}
        ),
        recent_records=lambda **_: records,
    )
    runtime = SimpleNamespace(
        client=client,
        identity=SimpleNamespace(backend="test"),
        _leases_lock=threading.Lock(),
        _leases={},
        _expire_leases=lambda: None,
        health=lambda: {"ok": True},
    )
    return dashboard.Dashboard(runtime)


def test_snapshot_includes_only_evidence_backed_edges_between_visible_records():
    from um_mem.contracts import EvidenceRef
    from um_mem.memory.backends.contracts import MemoryEdgeRecord

    records = [_panel_record(f"record-{index}") for index in range(3)]
    evidence = EvidenceRef(
        ref_type="test_run", ref_id="pytest:focused", produced_by="agent", producer_id="pytest"
    )
    valid = MemoryEdgeRecord(
        src_id="record-0",
        dst_id="record-1",
        edge_type="supports",
        workspace_id="ws",
        reason="Same reproduced parser failure.",
        weight=0.8,
        evidence_refs=(evidence,),
    )
    no_evidence = MemoryEdgeRecord(
        src_id="record-1", dst_id="record-2", edge_type="related_to", workspace_id="ws"
    )
    foreign = MemoryEdgeRecord(
        src_id="record-0",
        dst_id="record-2",
        edge_type="supports",
        workspace_id="another-workspace",
        evidence_refs=(evidence,),
    )
    edges = {
        "record-0": [valid, foreign],
        "record-1": [valid, no_evidence],
        "record-2": [no_evidence, foreign],
    }

    snapshot = _panel_for_records(records, edges).snapshot()

    assert snapshot["record_edges"] == [
        {
            "src_id": "record-0",
            "dst_id": "record-1",
            "edge_type": "supports",
            "reason": "Same reproduced parser failure.",
            "reason_truncated": False,
            "weight": 0.8,
            "evidence_refs": [
                {
                    "ref_type": "test_run",
                    "ref_id": "pytest:focused",
                    "hash": "",
                    "produced_by": "agent",
                    "producer_id": "pytest",
                }
            ],
            "evidence_ref_count": 1,
            "evidence_refs_truncated": False,
        }
    ]
    assert snapshot["record_edges_truncated"] is False
    assert snapshot["record_graph_scope"] == "recent_records"
    json.loads(json.dumps(snapshot))


def test_snapshot_caps_dense_record_graph_json():
    from um_mem.contracts import EvidenceRef
    from um_mem.memory.backends.contracts import MemoryEdgeRecord

    records = [_panel_record(f"record-{index:03d}") for index in range(64)]
    adjacency = {record.block_id: [] for record in records}
    evidence = tuple(
        EvidenceRef(
            ref_type="test_run",
            ref_id=f"pytest:dense-graph:{index}:" + "r" * 500,
            hash="h" * 500,
            produced_by="agent",
            producer_id="pytest-" + "p" * 500,
        )
        for index in range(4)
    )
    for index, source in enumerate(records):
        for target in records[index + 1 :]:
            edge = MemoryEdgeRecord(
                src_id=source.block_id,
                dst_id=target.block_id,
                edge_type="supports",
                workspace_id="ws",
                reason="Shared test evidence.",
                evidence_refs=evidence,
            )
            adjacency[source.block_id].append(edge)
            adjacency[target.block_id].append(edge)

    snapshot = _panel_for_records(records, adjacency).snapshot()
    encoded = json.dumps(snapshot, separators=(",", ":")).encode("utf-8")

    assert 0 < len(snapshot["record_edges"]) < 128
    assert snapshot["record_edges_truncated"] is True
    assert snapshot["record_graph_scope_limit"] == 64
    assert snapshot["record_graph_edge_limit"] == 128
    assert snapshot["record_graph_edge_byte_limit"] == 64 * 1024
    assert len(json.dumps(snapshot["record_edges"]).encode("utf-8")) <= 64 * 1024
    assert len(encoded) < 100_000


def test_snapshot_marks_bounded_catalog_edge_preview():
    snapshot = _panel_for_records(
        [],
        {},
        {"blocks": [], "edges": [{"src_id": "a", "dst_id": "b"}], "edges_truncated": True},
    ).snapshot()

    assert snapshot["block_edges_truncated"] is True
    assert snapshot["block_edge_limit"] == 128


def test_panel_project_scope_is_exact_for_snapshot_and_record_pages():
    calls = {}

    def recent_records(**kwargs):
        calls["recent"] = kwargs
        return []

    def list_records(**kwargs):
        calls["list"] = kwargs
        return {"records": [], "next_cursor": "next"}

    client = SimpleNamespace(
        workspace_id="ws",
        profile_id="profile-a",
        backend=SimpleNamespace(edges_between=lambda *_args, **_kwargs: []),
        catalog_snapshot=lambda **_: {"blocks": [], "edges": [], "edges_truncated": False},
        recent_records=recent_records,
        list_records=list_records,
        read_exact_records=lambda _ids: [],
    )
    runtime = SimpleNamespace(
        client=client,
        identity=SimpleNamespace(backend="test"),
        _leases_lock=threading.Lock(),
        _leases={},
        _expire_leases=lambda: None,
        health=lambda: {"ok": True},
    )
    panel = dashboard.Dashboard(runtime)
    selected_id = "group/one?x=1&y=2"

    snapshot = panel.snapshot(scope="project", scope_id=selected_id)
    records = panel.records("", "", scope="project", scope_id=selected_id)

    assert snapshot["record_scope"] == "project"
    assert snapshot["record_scope_id"] == selected_id
    assert calls["recent"]["scope"] == "project"
    assert calls["recent"]["scope_id"] == selected_id
    assert calls["list"]["scope"] == "project"
    assert calls["list"]["scope_id"] == selected_id
    assert records["next_cursor"] == "next"
    with pytest.raises(ValueError, match="exact project scope ID"):
        panel.records("", "", scope="project", scope_id=" ")
    with pytest.raises(ValueError, match="does not match"):
        panel.snapshot(scope="workspace", scope_id="other")


def test_project_record_list_discloses_recent_only_fallback_after_snapshot_cap():
    calls = {}
    records = [_panel_record(f"recent-{index}") for index in range(3)]

    def list_records(**kwargs):
        calls["list"] = kwargs
        raise ValueError("list_records scope exceeds 2048 records; select a narrower scope")

    def recent_records(**kwargs):
        calls["recent"] = kwargs
        return records

    client = SimpleNamespace(
        workspace_id="ws",
        profile_id="profile-a",
        list_records=list_records,
        recent_records=recent_records,
        read_exact_records=lambda _ids: [],
    )
    runtime = SimpleNamespace(client=client)
    panel = dashboard.Dashboard(runtime)

    result = panel.records("", "", scope="project", scope_id="project-a")

    assert result["scope_limited"] is True
    assert result["scope_limit"] == 2048
    assert result["recent_limit"] == 64
    assert result["next_cursor"] == ""
    assert calls["recent"]["scope"] == "project"
    assert calls["recent"]["scope_id"] == "project-a"


def test_sqlite_edges_between_bounds_results_and_scopes_workspace(tmp_path):
    from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex

    database = tmp_path / "graph.sqlite"
    backend = BackendSQLiteIndex(database, backend_id="test", workspace_id="ws")
    try:

        def insert_edge(selected_backend, src, dst, edge_type="supports", *, evidence=True):
            payload = json.dumps(
                {
                    "src_id": src,
                    "dst_id": dst,
                    "edge_type": edge_type,
                    "workspace_id": selected_backend.workspace_id,
                    "reason": "evidence",
                    "weight": 0.8,
                    "evidence_refs": [
                        {
                            "ref_type": "test_run",
                            "ref_id": "pytest:bounded-edges",
                            "produced_by": "agent",
                            "producer_id": "panel-test",
                        }
                    ]
                    if evidence
                    else [],
                    "metadata": {},
                }
            )
            selected_backend._conn.execute(
                "INSERT INTO edges(src_id,dst_id,edge_type,workspace_id,payload_json) "
                "VALUES(?,?,?,?,?)",
                (src, dst, edge_type, selected_backend.workspace_id, payload),
            )

        insert_edge(backend, "visible-a", "visible-b")
        insert_edge(backend, "visible-a", "visible-b", "aaa", evidence=False)
        insert_edge(backend, "visible-a", "outside-record")
        for index in range(200):
            insert_edge(backend, "visible-a", f"outside-{index:03d}")
        foreign_payload = json.dumps(
            {
                "src_id": "visible-a",
                "dst_id": "visible-b",
                "edge_type": "supports",
                "workspace_id": "other",
                "reason": "foreign",
                "weight": 1.0,
                "evidence_refs": [
                    {
                        "ref_type": "test_run",
                        "ref_id": "foreign",
                        "produced_by": "agent",
                        "producer_id": "foreign-test",
                    }
                ],
                "metadata": {},
            }
        )
        backend._conn.execute(
            "INSERT INTO edges(src_id,dst_id,edge_type,workspace_id,payload_json) "
            "VALUES(?,?,?,?,?)",
            ("visible-a", "visible-b", "supports", "other", foreign_payload),
        )
        backend._conn.commit()

        edges = backend.edges_between(("visible-a", "visible-b"), limit=129)
        plan = backend._conn.execute(
            "EXPLAIN QUERY PLAN SELECT payload_json FROM edges "
            "WHERE workspace_id=? AND src_id IN (?,?) AND dst_id IN (?,?) "
            "AND json_array_length(payload_json, '$.evidence_refs') > 0 "
            "ORDER BY src_id,dst_id,edge_type LIMIT ?",
            ("ws", "visible-a", "visible-b", "visible-a", "visible-b", 129),
        ).fetchall()
        assert [(edge.src_id, edge.dst_id) for edge in edges] == [("visible-a", "visible-b")]
        assert [edge.edge_type for edge in edges] == ["supports"]
        assert any("sqlite_autoindex_edges_1" in str(row) for row in plan)
        assert not any("TEMP B-TREE" in str(row).upper() for row in plan)
    finally:
        backend.close()


def test_correction_compares_expected_source_inside_dispatch(tmp_path):
    from um_mem import MemoryClient
    from um_mem.memory.backends.contracts import CanonicalHeadConflict
    from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex

    backend = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend)
    try:
        first = client.remember("First version", requested_block_id="stable_project_facts")
        original = client.read_exact_record(record_id=first.memory_id)
        changed = client.amend(
            original.block_id,
            expected_revision=original.revision_id,
            content="Other client's edit",
            reason="concurrent edit",
        )
        with pytest.raises(CanonicalHeadConflict, match="source revision"):
            client.correction(
                original.block_id,
                "Stale proposal",
                expected_revision=original.revision_id,
                requested_block_id="stable_project_facts",
            )
        assert (
            client.read_exact_record(record_id=changed["new_id"]).content == "Other client's edit"
        )
    finally:
        backend.close()


def test_trusted_correction_keeps_advisory_proposal_and_exact_source(tmp_path):
    from test_recoverable_correction import correction_setup

    client, backend, target, _, _ = correction_setup(tmp_path)
    try:
        result = client.correction(
            target.block_id,
            "Operator's proposed clarification",
            expected_revision=target.revision_id,
            requested_block_id="stable_project_facts",
        )
        assert result["pending_host_approval"] is True
        candidate = client.read_exact_record(record_id=result["block_id"])
        assert candidate.status == "candidate"
        assert candidate.authority == "advisory"
        assert client.read_exact_record(record_id=target.block_id).content == target.content
    finally:
        backend.close()


def test_replayed_save_keeps_record_identity(monkeypatch):
    panel = dashboard.Dashboard(SimpleNamespace(identity=object()))
    monkeypatch.setattr(
        panel.pool,
        "call",
        lambda *args, **kwargs: {
            "ok": True,
            "saved": True,
            "replayed": True,
            "canonical_reference": "mem-1",
        },
    )
    result = panel.call(
        "session",
        {
            "tool": "remember",
            "arguments": {
                "title": "Title",
                "content": "Content",
                "requested_block_id": "topic",
                "operation_id": "stable-save-id",
            },
        },
    )
    assert result["memory_id"] == "mem-1"


@pytest.mark.parametrize("command", ["dashboard", "map"])
def test_cli_opens_live_panel_without_generating_html(tmp_path, monkeypatch, capsys, command):
    from um_mem.memory.cli import main

    calls = []

    def open_panel(*args):
        calls.append(args)
        return {"ok": True, "url": "http://127.0.0.1:1234/panel/#test"}

    monkeypatch.setattr(dashboard, "dashboard_url", open_panel)
    assert (
        main(
            [
                command,
                "--repo-root",
                str(tmp_path),
                "--workspace",
                "test",
                "--profile",
                "default",
                "--backend",
                "mnemosyne",
                "--json",
                "--no-open",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["url"].endswith("/panel/#test")
    assert calls == [(str(tmp_path), "test", "default", "mnemosyne")]
    assert not list(tmp_path.glob("*.html"))


def test_cli_keeps_new_panel_alive_after_command_exits(tmp_path):
    from um_mem.memory.local_service import (
        build_local_service_identity,
        local_service_registry,
        shutdown_local_service_listener,
    )

    env = os.environ.copy()
    env["UM_MEM_CONFIG"] = str(tmp_path / "missing-config.json")
    env["UM_MEM_SEMANTIC"] = "0"
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="shared-test",
        profile_id="default",
        backend="mnemosyne",
        caller_role="operator",
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "um_mem.memory.cli",
                "dashboard",
                "--repo-root",
                str(tmp_path),
                "--workspace",
                "shared-test",
                "--profile",
                "default",
                "--backend",
                "mnemosyne",
                "--no-open",
                "--json",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        result = json.loads(completed.stdout)
        assert result["ok"] is True
        assert local_service_registry(identity) is not None
        with urllib.request.urlopen(result["url"].split("#", 1)[0], timeout=5) as response:
            assert response.status == 200
        assert dashboard.dashboard_url(
            str(tmp_path), "shared-test", "default", "mnemosyne", stop=True
        )["ok"]
        with pytest.raises(urllib.error.HTTPError) as stopped:
            urllib.request.urlopen(result["url"].split("#", 1)[0], timeout=5)
        assert stopped.value.code == 410
    finally:
        shutdown_local_service_listener(identity)
