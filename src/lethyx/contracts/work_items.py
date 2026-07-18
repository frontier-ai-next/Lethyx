"""Runtime-owned execute WorkItems and repair-subtask materialization."""

import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping

from lethyx.contracts.hashing import hash_value
from lethyx.contracts.models import json_ready

WorkItemKind = Literal[
    "implementation_repair",
    "proof_contract_repair",
    "oracle_repair",
    "packaging_import_repair",
    "env_capability_repair",
    "runtime_smoke_repair",
    "test_tampering_review",
]

WORK_ITEM_QUEUE_FILENAME = "work_items.json"
ACTIVE_WORK_ITEM_FILENAME = "active_work_item.json"
COMPLETED_WORK_ITEM_HISTORY_FILENAME = "completed_work_items.json"

WRITE_TOOLS = frozenset(
    {
        "apply_workspace_patch",
        "replace_workspace_file",
        "delete_workspace_file",
        "provision_workspace_environment",
    }
)
COMPLETION_TOOLS = frozenset({"mark_subtask_complete", "run_subtask_proof"})
PLAN_REPAIR_TOOLS = frozenset({"apply_plan_revision_patch"})
BLOCKED_CONTROL_TOOLS = frozenset(
    {
        "apply_workspace_patch",
        "replace_workspace_file",
        "delete_workspace_file",
        "provision_workspace_environment",
        "run_subtask_proof",
        "mark_subtask_complete",
        "request_watcher_review",
        "apply_plan_revision_patch",
    }
)
PACKAGING_IMPORT_FILES = (
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "workspace.toml",
)
_TEXT_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]+)"
)
_TEXT_TOP_LEVEL_FILE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])((?:README|CHANGELOG|LICENSE)\.md|pyproject\.toml|"
    r"package\.json|pytest\.ini|setup\.cfg|workspace\.toml)(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_TEXT_PATH_TOPS = {
    "src",
    "tests",
    "docs",
    "scripts",
    "assets",
    "backend",
    "frontend",
}
_TEXT_PATH_SUFFIXES = {
    ".py",
    ".css",
    ".html",
    ".js",
    ".jsx",
    ".txt",
    ".ts",
    ".tsx",
    ".md",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
}


@dataclass(frozen=True)
class WorkItem:
    id: str
    kind: WorkItemKind
    source_phase: str
    target_phase: str
    active_subtask_id: str = ""
    required_changes: tuple[dict[str, Any], ...] = ()
    allowed_files: tuple[str, ...] = ()
    forbidden_files: tuple[str, ...] = ()
    proof_contract: dict[str, Any] = field(default_factory=dict)
    execution_environment: dict[str, Any] = field(default_factory=dict)
    memory_scope: dict[str, Any] = field(default_factory=dict)
    tool_envelope: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[dict[str, Any], ...] = ()
    created_from_decision_id: str = ""
    attempt_id: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkItem":
        return cls(
            id=str(value.get("id") or ""),
            kind=_work_item_kind(value.get("kind")),
            source_phase=str(value.get("source_phase") or ""),
            target_phase=str(value.get("target_phase") or ""),
            active_subtask_id=str(value.get("active_subtask_id") or ""),
            required_changes=tuple(
                dict(item) for item in value.get("required_changes") or () if isinstance(item, dict)
            ),
            allowed_files=tuple(_unique_paths(value.get("allowed_files"))),
            forbidden_files=tuple(_unique_paths(value.get("forbidden_files"))),
            proof_contract=dict(value.get("proof_contract") or {})
            if isinstance(value.get("proof_contract"), dict)
            else {},
            execution_environment=dict(value.get("execution_environment") or {})
            if isinstance(value.get("execution_environment"), dict)
            else {},
            memory_scope=dict(value.get("memory_scope") or {})
            if isinstance(value.get("memory_scope"), dict)
            else {},
            tool_envelope=dict(value.get("tool_envelope") or {})
            if isinstance(value.get("tool_envelope"), dict)
            else {},
            evidence_refs=tuple(
                dict(item) for item in value.get("evidence_refs") or () if isinstance(item, dict)
            ),
            created_from_decision_id=str(value.get("created_from_decision_id") or ""),
            attempt_id=str(value.get("attempt_id") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source_phase": self.source_phase,
            "target_phase": self.target_phase,
            "active_subtask_id": self.active_subtask_id,
            "required_changes": [json_ready(item) for item in self.required_changes],
            "allowed_files": list(self.allowed_files),
            "forbidden_files": list(self.forbidden_files),
            "proof_contract": json_ready(self.proof_contract),
            "execution_environment": json_ready(self.execution_environment),
            "memory_scope": json_ready(self.memory_scope),
            "tool_envelope": json_ready(self.tool_envelope),
            "evidence_refs": [json_ready(item) for item in self.evidence_refs],
            "created_from_decision_id": self.created_from_decision_id,
            "attempt_id": self.attempt_id,
        }


def _work_item_kind(value: Any) -> WorkItemKind:
    text = str(value or "").strip()
    allowed = {
        "implementation_repair",
        "proof_contract_repair",
        "oracle_repair",
        "packaging_import_repair",
        "env_capability_repair",
        "runtime_smoke_repair",
        "test_tampering_review",
    }
    return text if text in allowed else "implementation_repair"  # type: ignore[return-value]


def _state_path(drive_root: Path, filename: str) -> Path:
    path = Path(drive_root) / "state" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_work_item_queue(drive_root: Path | None) -> list[WorkItem]:
    if drive_root is None:
        return []
    path = _state_path(Path(drive_root), WORK_ITEM_QUEUE_FILENAME)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = payload.get("work_items") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    return [WorkItem.from_mapping(item) for item in raw if isinstance(item, dict)]


def load_completed_work_items(drive_root: Path | None) -> list[WorkItem]:
    if drive_root is None:
        return []
    path = _state_path(Path(drive_root), COMPLETED_WORK_ITEM_HISTORY_FILENAME)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = payload.get("work_items") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    return [WorkItem.from_mapping(item) for item in raw if isinstance(item, dict)]


def save_completed_work_items(drive_root: Path, work_items: list[WorkItem]) -> Path:
    path = _state_path(drive_root, COMPLETED_WORK_ITEM_HISTORY_FILENAME)
    path.write_text(
        json.dumps(
            {"work_items": [item.to_dict() for item in work_items[-200:]]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def has_completed_work_item_for_decision(drive_root: Path | None, decision_id: str) -> bool:
    expected = str(decision_id or "").strip()
    if not expected:
        return False
    return any(
        item.created_from_decision_id == expected for item in load_completed_work_items(drive_root)
    )


def _work_item_blocker_fingerprints(item: WorkItem | None) -> set[str]:
    if item is None:
        return set()
    out: set[str] = set()
    for change in item.required_changes:
        if not isinstance(change, Mapping):
            continue
        value = str(change.get("blocker_fingerprint") or "").strip()
        if value:
            out.add(value)
    return out


def _normalizable_blocker_payload(
    decision: Mapping[str, Any],
    *,
    change: Mapping[str, Any] | None = None,
    issue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def _pick(mapping: Mapping[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
        if not isinstance(mapping, Mapping):
            return {}
        picked: dict[str, Any] = {}
        for key in keys:
            value = mapping.get(key)
            if value not in (None, "", [], {}):
                picked[key] = value
        return picked

    stable_decision = _pick(
        decision,
        (
            "phase_id",
            "target_phase",
            "outcome",
            "verification_summary",
            "required_plan_changes",
            "plan_revision_patch",
        ),
    )
    stable_change = _pick(
        change,
        (
            "id",
            "kind",
            "reason_code",
            "trigger_code",
            "code",
            "source_issue_code",
            "message",
            "description",
            "summary",
            "target_subtask_id",
            "file",
            "file_path",
            "path",
            "allowed_files",
            "files_to_change",
            "proof_contract",
            "required_plan_changes",
            "plan_revision_patch",
        ),
    )
    stable_issue = _pick(
        issue,
        (
            "code",
            "reason_code",
            "message",
            "description",
            "summary",
            "target_subtask_id",
            "file",
            "file_path",
            "path",
        ),
    )
    return {
        "decision": stable_decision,
        "change": stable_change,
        "issue": stable_issue,
    }


def loopback_blocker_fingerprint(
    decision: Mapping[str, Any],
    *,
    change: Mapping[str, Any] | None = None,
    issue: Mapping[str, Any] | None = None,
) -> str:
    if str(decision.get("outcome") or "") != "loop_back":
        return ""
    payload = _normalizable_blocker_payload(decision, change=change, issue=issue)
    return "blocker:" + hash_value(json_ready(payload))[:20]


def has_completed_work_item_for_blocker(drive_root: Path | None, blocker_fingerprint: str) -> bool:
    expected = str(blocker_fingerprint or "").strip()
    if not expected:
        return False
    return any(
        expected in _work_item_blocker_fingerprints(item)
        for item in load_completed_work_items(drive_root)
    )


def has_open_work_item_for_blocker(drive_root: Path | None, blocker_fingerprint: str) -> bool:
    expected = str(blocker_fingerprint or "").strip()
    if not expected:
        return False
    active = load_active_work_item(drive_root)
    if expected in _work_item_blocker_fingerprints(active):
        return True
    return any(
        expected in _work_item_blocker_fingerprints(item)
        for item in load_work_item_queue(drive_root)
    )


def save_work_item_queue(drive_root: Path, work_items: list[WorkItem]) -> Path:
    path = _state_path(drive_root, WORK_ITEM_QUEUE_FILENAME)
    path.write_text(
        json.dumps(
            {"work_items": [item.to_dict() for item in work_items]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_active_work_item(drive_root: Path | None) -> WorkItem | None:
    if drive_root is None:
        return None
    path = _state_path(Path(drive_root), ACTIVE_WORK_ITEM_FILENAME)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return WorkItem.from_mapping(payload) if isinstance(payload, dict) else None


def save_active_work_item(drive_root: Path, work_item: WorkItem) -> Path:
    path = _state_path(drive_root, ACTIVE_WORK_ITEM_FILENAME)
    path.write_text(
        json.dumps(work_item.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    queue = load_work_item_queue(drive_root)
    by_id = {item.id: item for item in queue}
    by_id[work_item.id] = work_item
    save_work_item_queue(drive_root, list(by_id.values()))
    return path


def clear_active_work_item(drive_root: Path) -> None:
    path = _state_path(drive_root, ACTIVE_WORK_ITEM_FILENAME)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def complete_active_work_item(drive_root: Path, work_item_id: str) -> None:
    current = load_active_work_item(drive_root)
    queue_before = load_work_item_queue(drive_root)
    completed = next((item for item in queue_before if item.id == work_item_id), None)
    if completed is None and current is not None and current.id == work_item_id:
        completed = current
    queue = [item for item in queue_before if item.id != work_item_id]
    save_work_item_queue(drive_root, queue)
    if completed is not None:
        history = load_completed_work_items(drive_root)
        by_id = {item.id: item for item in history}
        by_id[completed.id] = completed
        save_completed_work_items(drive_root, list(by_id.values()))
    if current is not None and current.id == work_item_id:
        clear_active_work_item(drive_root)


def _norm_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/").lstrip("/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _string_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        out: list[str] = []
        for key in (
            "path",
            "file_path",
            "file",
            "target",
            "target_file",
            "target_path",
            "value",
            "name",
        ):
            if isinstance(value.get(key), str):
                out.append(str(value.get(key) or ""))
        for key in (
            "files",
            "allowed_files",
            "files_to_change",
            "files_to_create",
            "files_affected",
            "changed_files",
        ):
            out.extend(_string_items(value.get(key)))
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[str] = []
        for item in value:
            out.extend(_string_items(item))
        return out
    return [str(value)]


def _unique_paths(value: Any) -> list[str]:
    paths: list[str] = []
    for item in _string_items(value):
        norm = _norm_path(item)
        if not norm:
            continue
        if any(ch.isspace() for ch in str(item)) or any(
            sep in str(item) for sep in (",", ";", "|")
        ):
            extracted = _paths_from_text(item)
            if extracted:
                paths.extend(extracted)
                continue
        paths.append(norm)
    return list(dict.fromkeys(paths))


def _paths_from_text(value: Any) -> list[str]:
    text = _payload_text(value)
    paths: list[str] = []
    for match in _TEXT_PATH_RE.findall(text):
        path = _norm_path(match.rstrip(".,);:]'\"`"))
        top = path.split("/", 1)[0].lower()
        suffix = Path(path).suffix.lower()
        if top in _TEXT_PATH_TOPS and suffix in _TEXT_PATH_SUFFIXES:
            paths.append(path)
    for match in _TEXT_TOP_LEVEL_FILE_RE.findall(text):
        paths.append(_norm_path(match))
    return list(dict.fromkeys(paths))


def _subtask_allowed_files(subtask: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(subtask, Mapping):
        return []
    paths: list[str] = []
    paths.extend(_unique_paths(subtask.get("allowed_files")))
    for key in ("files_to_create", "files_to_change", "files_affected"):
        paths.extend(_unique_paths(subtask.get(key)))
    proof = subtask.get("proof")
    scope = proof.get("scope") if isinstance(proof, Mapping) else None
    if isinstance(scope, Mapping):
        paths.extend(_unique_paths(scope.get("changed_files_expected")))
        paths.extend(_unique_paths(scope.get("files_under_test")))
    return list(dict.fromkeys(paths))


def _path_in_scope(path: str, scope: list[str]) -> bool:
    norm = _norm_path(path)
    if not norm:
        return True
    for raw in scope:
        candidate = _norm_path(raw)
        if not candidate:
            continue
        if candidate.endswith("/") and norm.startswith(candidate):
            return True
        if norm == candidate:
            return True
    return False


def _target_subtask_id(value: Mapping[str, Any]) -> str:
    for key in (
        "active_subtask_id",
        "target_subtask_id",
        "subtask_id",
        "target_subtask",
        "target_id",
    ):
        text = str(value.get(key) or "").strip()
        if text:
            return text
    return ""


def _payload_text(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, str):
            parts.append(value)
        else:
            try:
                parts.append(json.dumps(json_ready(value), ensure_ascii=False))
            except Exception:
                parts.append(str(value or ""))
    return "\n".join(part for part in parts if part)


def _text_mentions_mutation_smoke(value: Any) -> bool:
    text = _payload_text(value).lower()
    return "mutation_smoke" in text or "mutation survivor" in text or "mutant" in text


def _completed_desktop_gui_runtime_proof_present(
    subtasks: list[Mapping[str, Any]] | None,
) -> bool:
    for subtask in subtasks or []:
        if not isinstance(subtask, Mapping):
            continue
        status = str(subtask.get("status") or "").strip().lower()
        if status not in {"done", "complete", "completed"}:
            continue
        proof = subtask.get("proof")
        if not isinstance(proof, Mapping):
            continue
        harness = str(proof.get("harness_profile") or "").strip()
        required_caps = set(_string_items(proof.get("required_capabilities")))
        if harness == "desktop_gui_runtime" or "desktop_gui_runtime" in required_caps:
            return True
    return False


def _requests_missing_desktop_gui_runtime(value: Any) -> bool:
    text = _payload_text(value).lower().replace("-", " ")
    satisfied_context = (
        "runtime smoke test verified",
        "runtime smoke test passed",
        "desktop gui runtime proof passed",
        "desktop gui runtime proof verified",
        "gui interactions work",
        "ui wiring verified",
    )
    explicit_missing = (
        "missing desktop gui runtime",
        "missing desktop_gui_runtime",
        "no desktop gui runtime",
        "no desktop_gui_runtime",
        "without desktop gui runtime",
        "without desktop_gui_runtime",
    )
    if any(phrase in text for phrase in satisfied_context) and not any(
        phrase in text for phrase in explicit_missing
    ):
        return False
    if "desktop gui runtime" in text or "desktop_gui_runtime" in text:
        return True
    if "gui" not in text and "tkinter" not in text and "native ui" not in text:
        return False
    runtime_terms = (
        "run_real_e2e",
        "real e2e",
        "end to end",
        "e2e",
        "runtime",
        "interaction",
        "button",
        "launch",
        "managed runtime",
    )
    missing_terms = (
        "no ",
        "missing",
        "implement",
        "invoke",
        "required",
        "must",
    )
    return any(term in text for term in runtime_terms) and any(
        term in text for term in missing_terms
    )


def _is_alternative_repair_change(change: Mapping[str, Any]) -> bool:
    if bool(change.get("optional")):
        return True
    marker_fields = (
        "change_type",
        "mode",
        "kind",
        "relationship",
        "priority",
    )
    for marker_field in marker_fields:
        if str(change.get(marker_field) or "").strip().lower() in {
            "alternative",
            "alternate",
            "optional",
        }:
            return True
    text_fields = ("description", "message", "summary", "title")
    for text_field in text_fields:
        text = str(change.get(text_field) or "").strip().lower()
        if not text:
            continue
        if text.startswith(("alternatively", "alternative:", "alternate:")):
            return True
    return False


def _is_nonblocking_optional_repair_change(
    change: Mapping[str, Any],
    paired_issue: Mapping[str, Any] | None = None,
) -> bool:
    """Return True for advisory notes that should not mint execute WorkItems."""

    structured_keys = {
        "contract_path",
        "execution_environment",
        "invalid_values",
        "plan_revision_patch",
        "proof_contract",
        "required_deltas",
        "target_path",
    }
    issue = paired_issue if isinstance(paired_issue, Mapping) else {}
    for payload in (change, issue):
        if any(payload.get(key) for key in structured_keys):
            return False
        if str(payload.get("severity") or "").strip().lower() in {
            "blocking",
            "error",
            "human_required",
        }:
            return False
    if _text_mentions_mutation_smoke((change, issue)):
        return False
    if bool(change.get("optional")):
        return True
    marker = (
        str(
            change.get("change_type")
            or change.get("mode")
            or change.get("relationship")
            or change.get("priority")
            or ""
        )
        .strip()
        .lower()
    )
    if marker in {"advisory", "non_blocking", "non-blocking", "nice_to_have"}:
        return True
    text = " ".join(
        str(change.get(field) or "").strip().lower()
        for field in ("description", "message", "summary", "title")
    ).strip()
    return text.startswith(
        (
            "optional:",
            "optionally",
            "nice-to-have",
            "non-blocking",
            "advisory:",
        )
    )


_STRUCTURED_KIND_CODES: dict[str, WorkItemKind] = {
    "bad_generated_oracle": "oracle_repair",
    "bad_oracle": "oracle_repair",
    "oracle_repair": "oracle_repair",
    "package_import_env_mismatch": "packaging_import_repair",
    "packaging_import_repair": "packaging_import_repair",
    "setup_harness_mismatch": "packaging_import_repair",
    "proof_contract_repair": "proof_contract_repair",
    "plan_contract_revision": "proof_contract_repair",
    "invalid_plan_contract": "proof_contract_repair",
    "invalid_recovery_contract": "proof_contract_repair",
    "proof_execution_env_mismatch": "proof_contract_repair",
    "proof_execution_infra": "runtime_smoke_repair",
    "capability_probe_failed": "env_capability_repair",
    "capability_probe_environment_mismatch": "env_capability_repair",
    "dependency_provision_required": "env_capability_repair",
    "runtime_smoke_repair": "runtime_smoke_repair",
    "weak_runtime_proof": "runtime_smoke_repair",
    "test_tampering": "test_tampering_review",
    "test_tampering_review": "test_tampering_review",
}


def _payload_requests_plan_revision(value: Mapping[str, Any]) -> bool:
    if bool(value.get("requires_plan_patch")):
        return True
    if isinstance(value.get("plan_revision_patch"), Mapping):
        return True
    for key in ("required_plan_changes",):
        items = value.get(key)
        if isinstance(items, (list, tuple)) and items:
            return True
    return False


def _kind_from_structured_value(value: Any) -> WorkItemKind | None:
    if not isinstance(value, Mapping):
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                found = _kind_from_structured_value(item)
                if found:
                    return found
        return None
    for key in (
        "kind",
        "reason_code",
        "trigger_code",
        "code",
        "verdict",
        "source_issue_code",
    ):
        text = str(value.get(key) or "").strip()
        if text in _STRUCTURED_KIND_CODES:
            if text == "proof_execution_infra" and _payload_requests_plan_revision(value):
                return "proof_contract_repair"
            return _STRUCTURED_KIND_CODES[text]
    for key in (
        "recovery_decision",
        "issues",
        "required_changes",
        "required_plan_changes",
    ):
        found = _kind_from_structured_value(value.get(key))
        if found:
            return found
    return None


def classify_work_item_kind(*payloads: Any) -> WorkItemKind:
    for payload in payloads:
        found = _kind_from_structured_value(payload)
        if found:
            return found
    return "implementation_repair"


def _kind_from_control_payload(payload: Mapping[str, Any]) -> WorkItemKind:
    return classify_work_item_kind(payload)


def _repair_allowed_files(kind: WorkItemKind, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if kind == "packaging_import_repair":
        return PACKAGING_IMPORT_FILES
    return fallback


def _repair_forbidden_files(kind: WorkItemKind, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if kind == "packaging_import_repair":
        forbidden = set(fallback)
        forbidden.update(path for path in fallback if path.startswith(("src/", "tests/")))
        forbidden.update(("src/", "tests/"))
        return tuple(sorted(forbidden))
    return fallback


def reclassify_active_work_item(
    drive_root: Path,
    payload: Mapping[str, Any],
    *,
    decision_id: str = "",
) -> WorkItem | None:
    """Supersede the active WorkItem when proof evidence belongs to another lane."""

    current = load_active_work_item(drive_root)
    if current is None:
        return None
    kind = _kind_from_control_payload(payload)
    if kind == current.kind:
        return current
    text = _payload_text(payload)
    allowed_files = _repair_allowed_files(kind, current.allowed_files)
    forbidden_files = _repair_forbidden_files(kind, current.forbidden_files)
    required_change = {
        "id": str(payload.get("reason_code") or payload.get("code") or kind),
        "source": "WorkItemReclassifier",
        "reason_code": str(payload.get("reason_code") or payload.get("code") or ""),
        "message": text[:1200],
        "target_subtask_id": current.active_subtask_id,
        "evidence_refs": list(payload.get("evidence_refs") or []),
        "supersedes_work_item_id": current.id,
    }
    item_id = (
        "work:"
        + hash_value(
            {
                "supersedes": current.id,
                "kind": kind,
                "payload": json_ready(payload),
                "decision_id": decision_id,
                "attempt_id": current.attempt_id,
            }
        )[:16]
    )
    tool_envelope = _tool_envelope_for_kind(
        kind,
        allowed_files=list(allowed_files),
        forbidden_files=list(forbidden_files),
    )
    next_item = replace(
        current,
        id=item_id,
        kind=kind,
        required_changes=(required_change,),
        allowed_files=tuple(allowed_files),
        forbidden_files=tuple(forbidden_files),
        tool_envelope=tool_envelope,
        created_from_decision_id=decision_id or current.created_from_decision_id,
    )
    queue = [item for item in load_work_item_queue(drive_root) if item.id != current.id]
    queue.append(next_item)
    save_work_item_queue(drive_root, queue)
    save_active_work_item(drive_root, next_item)
    return next_item


def _module_name_from_text(text: str) -> str:
    for pattern in (
        r"No module named ['\"]([A-Za-z_][A-Za-z0-9_.]*)['\"]",
        r"\bimport\s+([A-Za-z_][A-Za-z0-9_.]*)\b",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).split(".")[0]
    return ""


# Simple-mutant labels emitted by an optional mutation-smoke verifier. A repair
# looped back from final_review's mutation_smoke must re-run that *same* gate, so
# we detect survivors from the loop-back text and which source files they touched.
_MUTATION_LABELS = (
    "gt_to_gte",
    "lt_to_lte",
    "and_to_or",
    "or_to_and",
    "true_to_false",
    "false_to_true",
)
_MUTATION_LABEL_SET = frozenset(_MUTATION_LABELS)


def _mutation_survivor_source_files(text: str) -> list[str]:
    """Source files with surviving simple mutants, parsed from loop-back text.

    The mutation_smoke gate emits survivors as ``path/to/file.py:<label>`` tokens
    (e.g. ``src/calc/engine.py:or_to_and``). We tokenize the text on the wrapping
    punctuation it appears in (quotes/brackets/commas/whitespace) and keep tokens
    that resolve to a production ``.py`` file paired with a known mutant label —
    plain string parsing, no semantic regex.
    """
    raw = str(text or "")
    for delimiter in "'\"[](){},":
        raw = raw.replace(delimiter, " ")
    seen: list[str] = []
    for token in raw.split():
        path_part, sep, label = token.rpartition(":")
        if not sep or label not in _MUTATION_LABEL_SET:
            continue
        if not path_part.endswith(".py"):
            continue
        path = path_part.replace("\\", "/").strip().strip("./")
        parts = [part.lower() for part in path.split("/") if part]
        # Mutation smoke only mutates production source, never test/doc files.
        if not path or not parts or parts[0] in {"tests", "test", "docs", "doc", ".memory"}:
            continue
        if path not in seen:
            seen.append(path)
    return seen


def _mutation_smoke_proof(
    *,
    source_files: list[str],
    scope_files: list[str],
    fallback: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Proof that re-runs the exact mutation_smoke gate over the survivor files.

    ``run_subtask_proof`` passes only on exit 0, so the repair subtask cannot be
    marked complete until the tests genuinely kill the mutants (a plain ``pytest``
    proof was strictly weaker than the gate and let repairs pass while the gate
    kept failing). ``PYTHONPATH`` is carried over from the failing subtask's proof
    so the workspace package still imports; ``run_subtask_proof`` additionally
    prepends the repo root so ``Lethyx`` is importable in-workspace.
    """
    command = ["python", "-m", "lethyx.verification.mutation_smoke"]
    for path in source_files:
        command.extend(["--file", path])
    env: dict[str, str] = {}
    if isinstance(fallback, Mapping):
        execution = fallback.get("execution")
        fallback_env = execution.get("env") if isinstance(execution, Mapping) else None
        if isinstance(fallback_env, Mapping):
            pythonpath = str(fallback_env.get("PYTHONPATH") or "").strip()
            if pythonpath:
                env["PYTHONPATH"] = pythonpath
    return {
        "execution": {
            "kind": "command",
            "command": command,
            "timeout_sec": 180,
            "env": env,
        },
        "oracle": {"required_properties": ["mutants_killed"]},
        "scope": {"changed_files_expected": scope_files or source_files},
        "harness_profile": "repair_smoke",
        "anti_gaming": {"requires_real_runtime": True},
    }


def _default_proof_contract(
    *,
    kind: WorkItemKind,
    text: str,
    fallback: Mapping[str, Any] | None,
    allowed_files: list[str],
) -> dict[str, Any]:
    if isinstance(fallback, Mapping) and fallback:
        return dict(fallback)
    if kind == "packaging_import_repair":
        module = _module_name_from_text(text) or "calculator"
        return {
            "execution": {
                "kind": "command",
                "command": ["python", "-c", f"import {module}"],
                "timeout_sec": 60,
            },
            "oracle": {"required_properties": ["module_imports"]},
            "scope": {"changed_files_expected": allowed_files},
            "harness_profile": "packaging_import",
        }
    if kind == "env_capability_repair":
        return {
            "execution": {
                "kind": "command",
                "command": [
                    "python",
                    "-c",
                    "import tkinter as tk; root=tk.Tk(); root.update(); root.destroy()",
                ],
                "timeout_sec": 60,
            },
            "oracle": {"required_properties": ["runtime_started"]},
            "scope": {"changed_files_expected": allowed_files},
            "required_capabilities": ["desktop_gui_runtime"],
            "harness_profile": "desktop_gui_runtime",
        }
    return {
        "execution": {
            "kind": "command",
            "command": ["python", "-m", "pytest", "tests", "-q"],
            "timeout_sec": 120,
            "env": {"PYTHONPATH": "src"},
        },
        "oracle": {"required_properties": ["tests_pass"]},
        "scope": {"changed_files_expected": allowed_files},
        "harness_profile": "repair_smoke",
    }


def _tool_envelope_for_kind(
    kind: WorkItemKind,
    *,
    allowed_files: list[str],
    forbidden_files: list[str],
    extra_allowed_tools: list[str] | None = None,
) -> dict[str, Any]:
    if kind == "proof_contract_repair":
        allowed = {
            "read_file",
            "list_files",
            "apply_plan_revision_patch",
            "run_subtask_proof",
            "mark_subtask_complete",
            "request_watcher_review",
        }
    else:
        allowed = {
            "read_file",
            "list_files",
            "apply_workspace_patch",
            "replace_workspace_file",
            "delete_workspace_file",
            "request_scope_change",
            "run_subtask_proof",
            "mark_subtask_complete",
            "request_watcher_review",
        }
        if kind in {
            "env_capability_repair",
            "runtime_smoke_repair",
            "packaging_import_repair",
        }:
            allowed.add("provision_workspace_environment")
    extras = {str(tool).strip() for tool in (extra_allowed_tools or []) if str(tool).strip()}
    if kind == "proof_contract_repair":
        extras &= allowed
    allowed.update(extras)
    return {
        "allowed_tools": sorted(allowed),
        "denied_tools": ["wipe_workspace", "reset_palace"],
        "policy": {
            "allowed_files": allowed_files,
            "forbidden_files": forbidden_files,
            "state_owner": "LETHYX_runtime",
        },
    }


def build_work_item_from_subtask(
    subtask: Mapping[str, Any],
    *,
    source_phase: str = "execute",
    target_phase: str = "execute",
    attempt_id: str = "",
    created_from_decision_id: str = "",
) -> WorkItem:
    subtask_id = str(subtask.get("id") or subtask.get("subtask_id") or "").strip()
    allowed_files = _subtask_allowed_files(subtask)
    proof = dict(subtask.get("proof") or {}) if isinstance(subtask.get("proof"), Mapping) else {}
    memory_scope = (
        dict(subtask.get("memory_scope") or {})
        if isinstance(subtask.get("memory_scope"), Mapping)
        else {}
    )
    extra_allowed_tools = _string_items(subtask.get("allowed_tools"))
    for asset in memory_scope.get("assets") or []:
        if isinstance(asset, Mapping) and str(asset.get("kind") or "") == "gmas_context":
            extra_allowed_tools.extend(["get_gmas_context", "search_gmas_knowledge"])
    item_id = (
        f"execute:{subtask_id}" if subtask_id else f"execute:{hash_value(json_ready(subtask))[:12]}"
    )
    tool_envelope = _tool_envelope_for_kind(
        "implementation_repair",
        allowed_files=allowed_files,
        forbidden_files=_unique_paths(subtask.get("forbidden_files")),
        extra_allowed_tools=extra_allowed_tools,
    )
    return WorkItem(
        id=item_id,
        kind="implementation_repair",
        source_phase=source_phase,
        target_phase=target_phase,
        active_subtask_id=subtask_id,
        allowed_files=tuple(allowed_files),
        forbidden_files=tuple(_unique_paths(subtask.get("forbidden_files"))),
        proof_contract=proof,
        memory_scope=memory_scope,
        tool_envelope=tool_envelope,
        created_from_decision_id=created_from_decision_id,
        attempt_id=attempt_id,
    )


def ensure_active_work_item_for_subtask(
    drive_root: Path,
    subtask: Mapping[str, Any],
    *,
    attempt_id: str = "",
) -> WorkItem:
    subtask_id = str(subtask.get("id") or subtask.get("subtask_id") or "").strip()
    expected = build_work_item_from_subtask(subtask, attempt_id=attempt_id)
    current = load_active_work_item(drive_root)
    if current is not None and current.active_subtask_id == subtask_id:
        if _work_item_needs_subtask_sync(current, expected):
            save_active_work_item(drive_root, expected)
            return expected
        return current
    for item in load_work_item_queue(drive_root):
        if item.active_subtask_id == subtask_id:
            if _work_item_needs_subtask_sync(item, expected):
                save_active_work_item(drive_root, expected)
                return expected
            save_active_work_item(drive_root, item)
            return item
    save_active_work_item(drive_root, expected)
    return expected


def _work_item_needs_subtask_sync(current: WorkItem, expected: WorkItem) -> bool:
    """Keep runtime WorkItems derived from the accepted subtask contract."""

    if current.proof_contract != expected.proof_contract:
        return True
    if current.memory_scope != expected.memory_scope:
        return True
    expected_allowed = set(expected.allowed_files)
    current_allowed = set(current.allowed_files)
    if expected_allowed and not expected_allowed.issubset(current_allowed):
        if current.kind == "packaging_import_repair" and current.required_changes:
            return False
        return True
    expected_forbidden = set(expected.forbidden_files)
    if expected_forbidden and expected_forbidden != set(current.forbidden_files):
        return True
    return False


def _evidence_refs_from_payload(*values: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key in ("evidence_refs", "proof_refs"):
            raw = value.get(key)
            if isinstance(raw, list):
                refs.extend(dict(item) for item in raw if isinstance(item, dict))
        ref = value.get("verification_report_ref")
        if isinstance(ref, Mapping):
            refs.append(
                {
                    "ref_type": "verification_report",
                    "ref_id": str(ref.get("report_id") or ""),
                    "hash": str(ref.get("ledger_hash") or ref.get("hash") or ""),
                    "produced_by": "verifier",
                }
            )
    return refs


def materialize_work_items_from_phase_exit(
    decision: Mapping[str, Any],
    *,
    execute_subtasks: list[Mapping[str, Any]] | None = None,
    attempt_id: str = "",
) -> list[WorkItem]:
    if str(decision.get("outcome") or "") != "loop_back":
        return []
    target_phase = str(decision.get("target_phase") or "execute").strip() or "execute"
    required_changes = [
        dict(item) for item in decision.get("required_changes") or () if isinstance(item, dict)
    ]
    issues = [dict(item) for item in decision.get("issues") or () if isinstance(item, dict)]
    if not required_changes and issues:
        required_changes = [
            {
                "id": str(issue.get("code") or f"issue-{idx}"),
                "source": "ReviewIssue",
                "reason_code": str(issue.get("code") or ""),
                "message": str(issue.get("message") or ""),
                "target_subtask_id": _target_subtask_id(issue),
                "evidence_refs": issue.get("evidence_refs") or [],
            }
            for idx, issue in enumerate(issues)
        ]
    if not required_changes:
        text = _payload_text(decision.get("verification_summary"), decision)
        if text.strip():
            required_changes = [
                {
                    "id": "verification-loopback",
                    "source": "PhaseExitDecision",
                    "message": text[:1200],
                    "evidence_refs": decision.get("evidence_refs") or [],
                }
            ]
    if not required_changes:
        return []

    if len(required_changes) > 1:
        primary_changes = [
            change for change in required_changes if not _is_alternative_repair_change(change)
        ]
        if primary_changes:
            required_changes = primary_changes

    completed_desktop_gui_runtime = _completed_desktop_gui_runtime_proof_present(execute_subtasks)
    if completed_desktop_gui_runtime:
        filtered_changes: list[dict[str, Any]] = []
        for idx, change in enumerate(required_changes):
            paired_issue = issues[idx] if idx < len(issues) else {}
            if _requests_missing_desktop_gui_runtime((change, paired_issue)):
                continue
            filtered_changes.append(change)
        required_changes = filtered_changes
        if not required_changes:
            return []

    subtasks_by_id = {
        str(item.get("id") or item.get("subtask_id") or ""): item
        for item in (execute_subtasks or [])
        if isinstance(item, Mapping)
    }
    # Tracks every repair-subtask id already present or minted in this batch so
    # successive loop-backs always allocate a *fresh pending* card instead of
    # colliding with an already-`done` repair id (which left execute with no
    # active subtask: "cannot start without an active WorkItem").
    used_subtask_ids: set[str] = {sid for sid in subtasks_by_id if sid}
    decision_id = phase_exit_decision_id(decision)
    work_items: list[WorkItem] = []
    for idx, change in enumerate(required_changes):
        paired_issue = issues[idx] if idx < len(issues) else {}
        target_subtask_id = _target_subtask_id(change)
        active_subtask = subtasks_by_id.get(target_subtask_id) if target_subtask_id else None
        text = _payload_text(change, paired_issue)
        if not text.strip():
            text = _payload_text(change, issues, decision.get("verification_summary"))
        if _is_nonblocking_optional_repair_change(change, paired_issue):
            continue
        mutation_text = text
        if _text_mentions_mutation_smoke((change, paired_issue)):
            mutation_text = _payload_text(text, decision.get("verification_summary"))
        kind = classify_work_item_kind(change, paired_issue, text)
        if kind == "runtime_smoke_repair" and _payload_requests_plan_revision(decision):
            reason_codes: list[str] = []
            for payload in (change, paired_issue):
                if not isinstance(payload, Mapping):
                    continue
                reason_codes.extend(
                    str(payload.get(key) or "")
                    for key in ("kind", "reason_code", "trigger_code", "code")
                )
            reason_text = " ".join(reason_codes)
            if "proof_execution_infra" in reason_text:
                kind = "proof_contract_repair"
        blocker_fingerprint = loopback_blocker_fingerprint(
            decision,
            change=change,
            issue=paired_issue,
        )
        allowed_files = _unique_paths(change.get("allowed_files"))
        if not allowed_files:
            allowed_files = _unique_paths(change)
        if not allowed_files:
            allowed_files = _paths_from_text(mutation_text)
        if not allowed_files:
            allowed_files = _subtask_allowed_files(active_subtask)
        if kind == "packaging_import_repair" and not allowed_files:
            allowed_files = [path for path in PACKAGING_IMPORT_FILES if path in text] or list(
                PACKAGING_IMPORT_FILES
            )
        forbidden_files = _unique_paths(change.get("forbidden_files"))
        if isinstance(change.get("proof_contract"), Mapping):
            fallback_proof = change.get("proof_contract")
        else:
            fallback_proof = (
                active_subtask.get("proof")
                if isinstance(active_subtask, Mapping)
                and isinstance(active_subtask.get("proof"), Mapping)
                else {}
            )
            if isinstance(fallback_proof, Mapping) and fallback_proof:
                subtask_scope = _subtask_allowed_files(active_subtask)
                concrete_allowed = [
                    path
                    for path in allowed_files
                    if path and not path.endswith("/") and "*" not in path
                ]
                if (
                    concrete_allowed
                    and subtask_scope
                    and any(not _path_in_scope(path, subtask_scope) for path in concrete_allowed)
                ):
                    fallback_proof = {}
        item_change: dict[str, Any] = dict(change)
        if blocker_fingerprint:
            item_change["blocker_fingerprint"] = blocker_fingerprint
        mutation_sources = _mutation_survivor_source_files(mutation_text)
        if mutation_sources:
            # Survivor loop-back: prove the fix against the *same* mutation gate
            # rather than inheriting the weaker pytest proof. The agent edits tests
            # to kill the mutant, so keep those files writable in scope.
            for source in mutation_sources:
                if source not in allowed_files:
                    allowed_files.append(source)
            # Killing a surviving mutant means strengthening the *tests*, so grant
            # write access to the tests tree up front instead of forcing the agent
            # through a request_scope_change round-trip (the stale bundle scope was
            # noisy and cost a loop). "tests/" is a directory-prefix scope pattern.
            if "tests/" not in allowed_files:
                allowed_files.append("tests/")
            proof_contract = _mutation_smoke_proof(
                source_files=mutation_sources,
                scope_files=mutation_sources,
                fallback=fallback_proof if isinstance(fallback_proof, Mapping) else {},
            )
            # Soft, data-driven guidance: tell the agent which gate failed, which
            # mutants survived, and that a green pytest is not the bar here.
            item_change["repair_guidance"] = (
                "final_review failed on mutation_smoke: simple logic mutants survived in "
                + ", ".join(mutation_sources)
                + ". A green `pytest` is NOT sufficient: the current tests do not "
                "distinguish correct behaviour from the mutated operator. Strengthen the "
                "tests so they FAIL when the operator is flipped (assert real computed "
                "outputs and exercise both branches). Do NOT weaken, skip, or delete tests, "
                "and do not merely add explanatory comments — the subtask proof re-runs the "
                "same mutation gate and only passes once the mutants are killed."
            )
        else:
            proof_contract = _default_proof_contract(
                kind=kind,
                text=text,
                fallback=fallback_proof if isinstance(fallback_proof, Mapping) else {},
                allowed_files=allowed_files,
            )
        repair_base = f"repair-{kind.replace('_', '-')}"
        # Allocate a unique repair-subtask id. Reusing an already-existing
        # (possibly `done`) id, or recursively re-prefixing
        # `repair-...-repair-...` across loop-backs, collided on the same id so
        # no new *pending* card was created and execute could not restart.
        fallback_id = target_subtask_id or f"{repair_base}-{idx + 1}"
        if fallback_id and fallback_id not in used_subtask_ids:
            repair_subtask_id = fallback_id
        else:
            counter = idx + 1
            repair_subtask_id = f"{repair_base}-{counter}"
            while repair_subtask_id in used_subtask_ids:
                counter += 1
                repair_subtask_id = f"{repair_base}-{counter}"
        used_subtask_ids.add(repair_subtask_id)
        item_seed = {
            "blocker_fingerprint": blocker_fingerprint or decision_id,
            "idx": idx,
            "kind": kind,
            "active_subtask_id": repair_subtask_id,
            "change": change,
        }
        item_id = f"work:{hash_value(item_seed)[:16]}"
        evidence_refs = _evidence_refs_from_payload(decision, change, *issues)
        tool_envelope = _tool_envelope_for_kind(
            kind,
            allowed_files=allowed_files,
            forbidden_files=forbidden_files,
        )
        work_items.append(
            WorkItem(
                id=item_id,
                kind=kind,
                source_phase=str(decision.get("phase_id") or ""),
                target_phase=target_phase,
                active_subtask_id=repair_subtask_id,
                required_changes=(item_change,),
                allowed_files=tuple(allowed_files),
                forbidden_files=tuple(forbidden_files),
                proof_contract=proof_contract,
                execution_environment=dict(change.get("execution_environment") or {})
                if isinstance(change.get("execution_environment"), Mapping)
                else {},
                memory_scope=dict(change.get("memory_scope") or {})
                if isinstance(change.get("memory_scope"), Mapping)
                else {},
                tool_envelope=tool_envelope,
                evidence_refs=tuple(evidence_refs),
                created_from_decision_id=decision_id,
                attempt_id=attempt_id or str(time.time()),
            )
        )
    return work_items


def phase_exit_decision_id(decision: Mapping[str, Any]) -> str:
    return str(decision.get("source_tool_call_id") or "") or hash_value(json_ready(decision))[:12]


def work_item_to_repair_subtask(work_item: WorkItem) -> dict[str, Any]:
    title = work_item.kind.replace("_", " ").title()
    change_text = _payload_text(*work_item.required_changes)[:1000]
    create_files: list[str] = []
    for change in work_item.required_changes:
        action = str(
            change.get("change_type") or change.get("action") or change.get("op") or ""
        ).lower()
        if action in {"create", "add"}:
            create_files.extend(_unique_paths(change))
    create_files = list(dict.fromkeys(create_files))
    # Directory-prefix ("tests/") and glob scope entries widen the write guard but
    # are not concrete files, so they must not leak into files_to_change/affected
    # (those feed completion's changed_files materialization, which expects real
    # paths).
    concrete_files = []
    if work_item.kind != "proof_contract_repair":
        concrete_files = [
            path for path in work_item.allowed_files if not path.endswith("/") and "*" not in path
        ]
    return {
        "id": work_item.active_subtask_id,
        "title": title,
        "goal": change_text or title,
        "allowed_files": list(work_item.allowed_files),
        "allowed_tools": list(work_item.tool_envelope.get("allowed_tools") or []),
        "allowed_skills": [],
        "proof": work_item.proof_contract,
        "memory_scope": work_item.memory_scope,
        "files_to_create": create_files,
        "files_to_change": concrete_files,
        "files_affected": concrete_files,
        "status": "pending",
    }


def work_item_tool_filter(
    tool_filter: Mapping[str, Any],
    *,
    work_item: WorkItem | None,
    control_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "allow": list(tool_filter.get("allow") or []),
        "deny": list(tool_filter.get("deny") or []),
        "required": list(tool_filter.get("required") or []),
        "completion_prerequisites": dict(tool_filter.get("completion_prerequisites") or {}),
    }
    gated = WRITE_TOOLS | COMPLETION_TOOLS | PLAN_REPAIR_TOOLS
    if work_item is None:
        payload["allow"] = [tool for tool in payload["allow"] if tool not in gated]
        payload["deny"] = sorted(set(payload["deny"]) | gated)
        return payload
    if (
        isinstance(control_decision, Mapping)
        and str(control_decision.get("kind") or "") == "blocked_no_valid_next_action"
    ):
        allowed_next = set(_string_items(control_decision.get("allowed_next_tools")))
        if not allowed_next:
            allowed_next = {
                tool for tool in _string_items(control_decision.get("required_next_tools")) if tool
            }
        payload["allow"] = sorted(set(payload["allow"]) & allowed_next)
        payload["deny"] = sorted(set(payload["deny"]) | (BLOCKED_CONTROL_TOOLS - allowed_next))
        payload["required"] = [tool for tool in payload["required"] if tool in allowed_next]
        return payload
    envelope = work_item.tool_envelope or {}
    allowed = set(payload["allow"])
    scoped_allowed = set(envelope.get("allowed_tools") or [])
    if scoped_allowed:
        allowed &= scoped_allowed
        allowed |= COMPLETION_TOOLS & scoped_allowed
    if work_item.kind != "proof_contract_repair":
        allowed -= PLAN_REPAIR_TOOLS
    payload["allow"] = sorted(allowed)
    payload["deny"] = sorted(set(payload["deny"]) | set(envelope.get("denied_tools") or []))
    return payload
