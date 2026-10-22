import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from um_mem.memory import mcp_call
from um_mem.memory.local_service import build_local_service_identity
from um_mem.memory.service_receipts import (
    claim_remember_receipt,
    complete_remember_receipt,
    service_receipt_path,
)


@pytest.mark.parametrize("tool", ["remember", "set_active_mask"])
@pytest.mark.parametrize("error_type", ["ValueError", "ConnectionError"])
def test_failed_mutation_is_closed_but_same_request_can_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, error_type: str
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace", caller_role="operator"
    )
    calls: list[dict[str, object]] = []
    failure = {
        "ok": False,
        "error": f"{error_type}: private source text",
        "error_type": error_type,
        "reason_code": "memory_operation_failed",
    }
    success = {
        "ok": True,
        "saved": True,
        "memory_id": "mem_one",
        "raw": {"canonical_key": "key", "revision_id": "rev_one", "recorded_at": "time"},
        "backend": "mempalace",
        "workspace_id": "alpha",
    }

    def call_tool(_tool: str, arguments: dict[str, object]) -> dict[str, object]:
        calls.append(arguments)
        return failure if len(calls) == 1 else success

    monkeypatch.setattr(mcp_call, "call_tool", call_tool)
    request = {
        "protocol_version": identity.protocol_version,
        "request_id": "attempt-1",
        "lease": "lease",
        "caller_role": "operator",
        "mutation_correlation_id": "stable-operation",
        "tool": tool,
        "arguments": {"content": "private source text"},
    }

    def dispatch(arguments: dict[str, object]) -> dict[str, object]:
        return mcp_call._dispatch_local_service_request(
            identity, json.dumps(arguments), client=object(), expected_lease="lease"
        )

    dispatch(request)
    with sqlite3.connect(service_receipt_path(tmp_path)) as connection:
        status, raw, created_at = connection.execute(
            "SELECT status,response_json,created_at FROM remember_receipts"
        ).fetchone()
    assert status == "failed"
    assert "private source text" not in raw
    assert json.loads(raw) == {
        "ok": False,
        "error_type": error_type,
        "reason_code": "memory_operation_failed",
        "error_sha256": hashlib.sha256(failure["error"].encode()).hexdigest(),
    }
    conflict = dispatch({**request, "arguments": {"content": "different"}})
    assert conflict["payload"]["reason_code"] == "mutation_receipt_conflict"
    assert conflict["payload"]["retryable"] is False
    assert len(calls) == 1  # Failed receipts still enforce the original fingerprint.
    dispatch({**request, "request_id": "attempt-2"})
    with sqlite3.connect(service_receipt_path(tmp_path)) as connection:
        row = connection.execute("SELECT status,created_at FROM remember_receipts").fetchone()
    assert row == ("committed", created_at)
    if tool == "remember":
        assert calls[0]["operation_id"] == calls[1]["operation_id"] == "stable-operation"
        assert calls[0]["_recorded_at"] == calls[1]["_recorded_at"] == created_at
    dispatch({**request, "request_id": "attempt-3"})
    assert len(calls) == 2


def test_failed_completion_cannot_overwrite_success_or_accept_untyped_payload(
    tmp_path: Path,
) -> None:
    arguments = {
        "workspace_id": "alpha",
        "backend": "mempalace",
        "caller_role": "operator",
        "protocol_version": 1,
        "mutation_correlation_id": "operation",
        "arguments": {},
    }
    claim_remember_receipt(tmp_path, **arguments)
    with pytest.raises(ValueError):
        complete_remember_receipt(tmp_path, **arguments, outcome="failed", response={"ok": True})
    success = {
        "receipt_kind": "mutation_response",
        "mutation_tool": "create_block",
        "response": {"ok": True},
    }
    complete_remember_receipt(tmp_path, **arguments, response=success)
    with pytest.raises(RuntimeError, match="response mismatch"):
        complete_remember_receipt(
            tmp_path,
            **arguments,
            outcome="failed",
            response={
                "ok": False,
                "error_type": "ValueError",
                "reason_code": "invalid",
                "error_sha256": "0" * 64,
            },
        )
    assert claim_remember_receipt(tmp_path, **arguments)[:2] == ("committed", success)
