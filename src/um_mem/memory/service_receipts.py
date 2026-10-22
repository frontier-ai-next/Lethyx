"""Durable replay receipts for authenticated local-service mutations."""

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

_CREATE = """
CREATE TABLE IF NOT EXISTS remember_receipts (
    identity_sha256 TEXT NOT NULL,
    request_id_sha256 TEXT NOT NULL,
    request_fingerprint_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(identity_sha256, request_id_sha256)
);
"""


def service_receipt_path(repo_root: str | Path) -> Path:
    return Path(repo_root) / ".UM-MEM" / "memory" / "service_receipts.sqlite"


def host_tool_mutation_correlation_id(source_id: str, tool_call_id: str) -> str:
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in (source_id, tool_call_id)
    ):
        raise ValueError("host tool mutation requires exact source and invocation identities")
    identity = json.dumps([source_id, tool_call_id], separators=(",", ":"))
    return "hermes-agent-retain:" + _sha256(identity)


def read_committed_mutation_receipt(
    repo_root: str | Path,
    *,
    workspace_id: str,
    profile_id: str,
    backend: str,
    caller_role: str,
    protocol_version: int,
    mutation_correlation_id: str,
) -> dict[str, object] | None:
    path = service_receipt_path(repo_root)
    if not path.is_file():
        return None
    identity = _identity_sha256(workspace_id, profile_id, backend, caller_role, protocol_version)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        row = connection.execute(
            "SELECT response_json FROM remember_receipts "
            "WHERE identity_sha256=? AND request_id_sha256=? AND status='committed'",
            (identity, _sha256(mutation_correlation_id)),
        ).fetchone()
    return _decode_response(row[0]) if row is not None else None


def read_mutation_receipt_status(
    repo_root: str | Path,
    *,
    workspace_id: str,
    profile_id: str,
    backend: str,
    caller_role: str,
    protocol_version: int,
    mutation_correlation_id: str,
) -> tuple[str, dict[str, object] | None] | None:
    """Inspect an existing scoped mutation receipt without claiming or retrying it."""

    path = service_receipt_path(repo_root)
    if not path.is_file():
        return None
    identity = _identity_sha256(workspace_id, profile_id, backend, caller_role, protocol_version)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        row = connection.execute(
            "SELECT status, response_json FROM remember_receipts "
            "WHERE identity_sha256=? AND request_id_sha256=?",
            (identity, _sha256(mutation_correlation_id)),
        ).fetchone()
    if row is None:
        return None
    status, response_json = str(row[0]), str(row[1])
    if status == "pending":
        return status, None
    if status in {"committed", "failed"}:
        return status, _decode_response(response_json)
    raise RuntimeError("local service remember receipt state invalid")


def claim_remember_receipt(
    repo_root: str | Path,
    *,
    workspace_id: str,
    profile_id: str = "default",
    backend: str,
    caller_role: str,
    protocol_version: int,
    mutation_correlation_id: str,
    arguments: dict[str, object],
) -> tuple[str, dict[str, object] | None, str]:
    path = service_receipt_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity_sha256 = _identity_sha256(
        workspace_id,
        profile_id,
        backend,
        caller_role,
        protocol_version,
    )
    mutation_correlation_sha256 = _sha256(mutation_correlation_id)
    fingerprint_sha256 = _fingerprint_sha256(arguments)
    now = datetime.now(UTC).isoformat()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(_CREATE)
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT request_fingerprint_sha256, status, response_json, created_at
            FROM remember_receipts
            WHERE identity_sha256=? AND request_id_sha256=?
            """,
            (identity_sha256, mutation_correlation_sha256),
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO remember_receipts(
                    identity_sha256, request_id_sha256, request_fingerprint_sha256,
                    status, response_json, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    identity_sha256,
                    mutation_correlation_sha256,
                    fingerprint_sha256,
                    "pending",
                    "",
                    now,
                    now,
                ),
            )
            return "new", None, now
        stored_fingerprint, status, response_json, created_at = (str(value) for value in row)
        if stored_fingerprint != fingerprint_sha256:
            raise ValueError("local service remember receipt conflict")
        if status == "committed":
            return "committed", _decode_response(response_json), created_at
        if status == "failed":
            if _decode_response(response_json).get("ok") is not False:
                raise RuntimeError("local service failed receipt response invalid")
            connection.execute(
                "UPDATE remember_receipts SET status='pending', updated_at=? "
                "WHERE identity_sha256=? AND request_id_sha256=? AND status='failed'",
                (now, identity_sha256, mutation_correlation_sha256),
            )
            return "pending", None, created_at
        if status == "pending":
            return "pending", None, created_at
        raise RuntimeError("local service remember receipt state invalid")


def complete_remember_receipt(
    repo_root: str | Path,
    *,
    workspace_id: str,
    profile_id: str = "default",
    backend: str,
    caller_role: str,
    protocol_version: int,
    mutation_correlation_id: str,
    arguments: dict[str, object],
    response: dict[str, object],
    outcome: Literal["committed", "failed"] = "committed",
) -> None:
    if outcome not in {"committed", "failed"}:
        raise ValueError("local service receipt outcome invalid")
    if outcome == "failed":
        required = {"ok", "error_type", "reason_code", "error_sha256"}
        if (
            set(response) != required
            or response.get("ok") is not False
            or any(not isinstance(response.get(key), str) for key in required - {"ok"})
            or len(str(response["error_sha256"])) != 64
        ):
            raise ValueError("local service failed receipt response invalid")
    path = service_receipt_path(repo_root)
    identity_sha256 = _identity_sha256(
        workspace_id,
        profile_id,
        backend,
        caller_role,
        protocol_version,
    )
    mutation_correlation_sha256 = _sha256(mutation_correlation_id)
    fingerprint_sha256 = _fingerprint_sha256(arguments)
    response_json = json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    now = datetime.now(UTC).isoformat()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(_CREATE)
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT request_fingerprint_sha256, status, response_json
            FROM remember_receipts
            WHERE identity_sha256=? AND request_id_sha256=?
            """,
            (identity_sha256, mutation_correlation_sha256),
        ).fetchone()
        if row is None:
            raise RuntimeError("local service remember receipt missing")
        stored_fingerprint, status, stored_response = (str(value) for value in row)
        if stored_fingerprint != fingerprint_sha256:
            raise ValueError("local service remember receipt conflict")
        if status == "committed":
            if outcome != "committed" or _decode_response(stored_response) != response:
                raise RuntimeError("local service remember receipt response mismatch")
            return
        if status != "pending":
            raise RuntimeError("local service remember receipt state invalid")
        connection.execute(
            """
            UPDATE remember_receipts
            SET status=?, response_json=?, updated_at=?
            WHERE identity_sha256=? AND request_id_sha256=?
            """,
            (
                outcome,
                response_json,
                now,
                identity_sha256,
                mutation_correlation_sha256,
            ),
        )


def _identity_sha256(
    workspace_id: str,
    profile_id: str,
    backend: str,
    caller_role: str,
    protocol_version: int,
) -> str:
    return _sha256(
        json.dumps(
            {
                "workspace_id": workspace_id,
                "profile_id": profile_id,
                "backend": backend,
                "caller_role": caller_role,
                "protocol_version": protocol_version,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _fingerprint_sha256(arguments: dict[str, object]) -> str:
    return _sha256(json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decode_response(raw: str) -> dict[str, object]:
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("local service remember receipt response invalid") from exc
    if not isinstance(response, dict):
        raise RuntimeError("local service remember receipt response invalid")
    if set(response) == {"ok", "error_type", "reason_code", "error_sha256"}:
        if (
            response.get("ok") is not False
            or any(
                not isinstance(response.get(key), str)
                for key in ("error_type", "reason_code", "error_sha256")
            )
            or len(response["error_sha256"]) != 64
        ):
            raise RuntimeError("local service failed receipt response invalid")
        return dict(response)
    if set(response) == {"receipt_kind", "mutation_tool", "response"}:
        mutation_tools = {
            "remember",
            "promote",
            "create_block",
            "amend_block",
            "move_block",
            "archive_block",
            "revive_block",
            "delete_block",
            "split_block",
            "merge_blocks",
            "add_block_edge",
            "remove_block_edge",
            "set_active_mask",
            "amend_active_mask",
            "transition_active_mask",
            "observe_maintenance",
            "consolidate_observations",
        }
        stored_response = response.get("response")
        if (
            response.get("receipt_kind") != "mutation_response"
            or response.get("mutation_tool") not in mutation_tools
            or not isinstance(stored_response, dict)
            or stored_response.get("ok") is not True
        ):
            raise RuntimeError("local service mutation receipt response invalid")
        return dict(response)
    success_required: dict[str, type | tuple[type, ...]] = {
        "ok": bool,
        "saved": bool,
        "canonical_reference": str,
        "canonical_key": str,
        "revision_id": str,
        "recorded_at": str,
        "native_reference": (str, type(None)),
        "backend": str,
        "workspace_id": str,
    }
    if set(response) == set(success_required):
        if any(
            not isinstance(response.get(key), value_type)
            for key, value_type in success_required.items()
        ):
            raise RuntimeError("local service remember receipt response invalid")
        if response["ok"] is not True or response["saved"] is not True:
            raise RuntimeError("local service remember receipt response invalid")
        return dict(response)
    observation_required = {
        "ok": bool,
        "saved": bool,
        "witness_id": str,
        "memory_id": str,
        "requested_block_id": str,
        "actual_block_id": str,
        "route_outcome": str,
        "reason": str,
        "status": str,
        "authority": str,
        "verification_status": str,
        "evidence_refs": list,
        "backend": str,
        "workspace_id": str,
    }
    observation_keys = set(response)
    source_fields = {"source_id", "source_sha256", "source_bytes"}
    if observation_keys in (
        set(observation_required),
        {*observation_required, "verification_report"},
        {*observation_required, *source_fields},
        {*observation_required, *source_fields, "verification_report"},
    ):
        if any(
            not isinstance(response.get(key), value_type)
            for key, value_type in observation_required.items()
        ):
            raise RuntimeError("local service observe receipt response invalid")
        verification_report = response.get("verification_report")
        if source_fields <= observation_keys and (
            not isinstance(response.get("source_id"), str)
            or not response["source_id"]
            or not isinstance(response.get("source_sha256"), str)
            or len(response["source_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in response["source_sha256"])
            or type(response.get("source_bytes")) is not int
            or response["source_bytes"] <= 0
        ):
            raise RuntimeError("local service observe source receipt invalid")
        if verification_report is not None and (
            not isinstance(verification_report, dict)
            or not {
                "report_id",
                "report_hash",
                "workspace_hash",
                "diff_hash",
                "produced_after_event_id",
                "verifier_id",
                "passed",
                "ledger_hash",
            }.issubset(verification_report)
        ):
            raise RuntimeError("local service observe receipt response invalid")
        if (
            response["ok"] is not True
            or response["saved"] is not True
            or not response["witness_id"]
            or not response["route_outcome"]
            or not response["reason"]
        ):
            raise RuntimeError("local service observe receipt response invalid")
        return dict(response)
    mutation_required = {
        "ok": bool,
        "saved": bool,
        "mutation_tool": str,
        "canonical_reference": str,
        "previous_revision": str,
        "revision": str,
        "archived": bool,
        "backend": str,
        "workspace_id": str,
    }
    if set(response) == set(mutation_required):
        if any(
            not isinstance(response.get(key), value_type)
            for key, value_type in mutation_required.items()
        ):
            raise RuntimeError("local service mutation receipt response invalid")
        if (
            response["ok"] is not True
            or response["saved"] is not True
            or response["mutation_tool"]
            not in {"amend", "create_block", "amend_block", "archive_block"}
            or not response["canonical_reference"]
            or not response["revision"]
            or not response["backend"]
            or not response["workspace_id"]
        ):
            raise RuntimeError("local service mutation receipt response invalid")
        return dict(response)
    rejected_required = {
        "ok": bool,
        "saved": bool,
        "route_rejected": bool,
        "reason": str,
        "backend": str,
        "workspace_id": str,
    }
    rejected_keys = set(response)
    rejected_optional = {"suggested_route", "route_decision", "policy_issues"}
    if set(rejected_required) <= rejected_keys <= set(rejected_required) | rejected_optional:
        if any(
            not isinstance(response.get(key), value_type)
            for key, value_type in rejected_required.items()
        ):
            raise RuntimeError("local service remember receipt response invalid")
        if (
            response["ok"] is not False
            or response["saved"] is not False
            or response["route_rejected"] is not True
            or not response["reason"]
        ):
            raise RuntimeError("local service remember receipt response invalid")
        route_decision = response.get("route_decision")
        if route_decision is not None and not isinstance(route_decision, dict):
            raise RuntimeError("local service remember receipt response invalid")
        policy_issues = response.get("policy_issues")
        if policy_issues is not None and (
            not isinstance(policy_issues, list)
            or any(not isinstance(issue, str) or not issue for issue in policy_issues)
        ):
            raise RuntimeError("local service remember receipt response invalid")
        suggested_route = response.get("suggested_route")
        if suggested_route is not None and (
            not isinstance(suggested_route, dict)
            or set(suggested_route) != {"action", "requested_block_id"}
            or any(
                not isinstance(suggested_route.get(key), str) or not suggested_route.get(key)
                for key in ("action", "requested_block_id")
            )
        ):
            raise RuntimeError("local service remember receipt response invalid")
        return dict(response)
    raise RuntimeError("local service remember receipt response invalid")
