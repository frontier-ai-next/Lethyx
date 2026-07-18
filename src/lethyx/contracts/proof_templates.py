"""Platform-owned canonical proof templates.

These templates keep common proof contracts valid-by-construction. Agents may
select a template with ``proof.template_id`` and provide narrow scope hints; the
compiler instantiates the concrete ProofSpec fields before validators run.
"""

from dataclasses import dataclass
from typing import Any


CANONICAL_PROOF_TEMPLATE_IDS = frozenset(
    {
        "python_src_unit",
        "desktop_gui_headless_model",
        "desktop_gui_runtime_smoke",
    }
)


@dataclass(frozen=True)
class ProofTemplate:
    id: str
    harness_profile: str
    summary: str


_TEMPLATES: dict[str, ProofTemplate] = {
    "python_src_unit": ProofTemplate(
        id="python_src_unit",
        harness_profile="python_src_layout",
        summary="Pytest proof for Python src-layout code with PYTHONPATH=src.",
    ),
    "desktop_gui_headless_model": ProofTemplate(
        id="desktop_gui_headless_model",
        harness_profile="desktop_gui_headless",
        summary="Headless pytest proof for native GUI model/controller behavior.",
    ),
    "desktop_gui_runtime_smoke": ProofTemplate(
        id="desktop_gui_runtime_smoke",
        harness_profile="desktop_gui_runtime",
        summary="Managed native GUI runtime smoke with readiness and cleanup.",
    ),
}


def known_proof_template_ids() -> tuple[str, ...]:
    return tuple(_TEMPLATES)


def canonical_proof_template_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for template_id in _TEMPLATES:
        if text == template_id:
            return template_id
        if text.startswith(template_id):
            suffix = text[len(template_id) :]
            if suffix and not (suffix[0].isalnum() or suffix[0] == "_"):
                return template_id
    return text


def get_proof_template(template_id: str) -> ProofTemplate | None:
    return _TEMPLATES.get(canonical_proof_template_id(template_id))


def _list_strings(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _unique_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for raw in paths:
        path = str(raw or "").replace("\\", "/").strip().strip("/")
        if not path or path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _proof_scope(raw_proof: dict[str, Any]) -> dict[str, Any]:
    scope = raw_proof.get("scope")
    return dict(scope) if isinstance(scope, dict) else {}


def _template_args(raw_proof: dict[str, Any]) -> dict[str, Any]:
    args = raw_proof.get("template_args")
    return dict(args) if isinstance(args, dict) else {}


def _subtask_paths(subtask: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("files_to_create", "files_to_change", "files_affected"):
        paths.extend(_list_strings(subtask.get(key)))
    scope = _proof_scope(dict(subtask.get("proof") or {}))
    for key in (
        "files_under_test",
        "changed_files_expected",
        "pytest_targets",
        "test_targets",
    ):
        paths.extend(_list_strings(scope.get(key)))
    return _unique_paths(paths)


def _infer_package(raw_proof: dict[str, Any], subtask: dict[str, Any]) -> str:
    args = _template_args(raw_proof)
    package = str(args.get("package") or raw_proof.get("package") or "").strip()
    if package:
        return package
    for path in _subtask_paths(subtask):
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "src" and parts[-1].endswith(".py"):
            return parts[1]
    return "app"


def _declared_paths(subtask: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("files_to_create", "files_to_change", "files_affected"):
        paths.extend(_list_strings(subtask.get(key)))
    return _unique_paths(paths)


def _production_paths(subtask: dict[str, Any], scope: dict[str, Any]) -> list[str]:
    explicit = _list_strings(scope.get("files_under_test"))
    if explicit:
        return explicit
    paths = []
    for path in _declared_paths(subtask):
        if path.endswith(".py") and not path.startswith("tests/"):
            paths.append(path)
    return paths


def _changed_paths(subtask: dict[str, Any], scope: dict[str, Any]) -> list[str]:
    explicit = _list_strings(scope.get("changed_files_expected"))
    if explicit:
        return _unique_paths([*explicit, *_declared_paths(subtask)])
    return _declared_paths(subtask)


def _runtime_changed_paths(
    raw_proof: dict[str, Any], subtask: dict[str, Any], scope: dict[str, Any]
) -> list[str]:
    changed = _changed_paths(subtask, scope)
    package = _infer_package(raw_proof, subtask)
    entrypoint = f"src/{package}/__main__.py"
    package_prefix = f"src/{package}/"
    package_paths = [path for path in _subtask_paths(subtask) if path.startswith(package_prefix)]
    if package_paths and entrypoint not in changed:
        changed.append(entrypoint)
    return _unique_paths(changed)


def _pytest_targets(subtask: dict[str, Any], scope: dict[str, Any]) -> list[str]:
    explicit = _list_strings(scope.get("pytest_targets") or scope.get("test_targets"))
    if explicit:
        return explicit
    tests = [
        path
        for path in _declared_paths(subtask)
        if path.startswith("tests/") and path.endswith(".py")
    ]
    return tests or ["tests"]


def _merge_env(base: dict[str, str], raw_execution: dict[str, Any]) -> dict[str, str]:
    merged = dict(base)
    raw_env = raw_execution.get("env")
    if isinstance(raw_env, dict):
        merged.update({str(key): str(value) for key, value in raw_env.items()})
    return merged


def _oracle_properties(raw_proof: dict[str, Any], defaults: list[str]) -> list[str]:
    oracle = raw_proof.get("oracle")
    raw_props = oracle.get("required_properties") if isinstance(oracle, dict) else None
    props = defaults + _list_strings(raw_props)
    seen: set[str] = set()
    unique: list[str] = []
    for prop in props:
        if prop not in seen:
            seen.add(prop)
            unique.append(prop)
    return unique


def _base_passthrough(raw_proof: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("generated_test_contract", "human_claims", "evidence_refs"):
        if key in raw_proof:
            result[key] = raw_proof[key]
    return result


def _template_scope(raw_proof: dict[str, Any], subtask: dict[str, Any]) -> dict[str, Any]:
    scope = _proof_scope(raw_proof)
    return {
        "files_under_test": _production_paths(subtask, scope),
        "changed_files_expected": _changed_paths(subtask, scope),
        "pytest_targets": _pytest_targets(subtask, scope),
    }


def _python_src_unit(raw_proof: dict[str, Any], subtask: dict[str, Any]) -> dict[str, Any]:
    raw_execution = dict(raw_proof.get("execution") or {})
    targets = _pytest_targets(subtask, _proof_scope(raw_proof))
    proof = {
        **_base_passthrough(raw_proof),
        "template_id": "python_src_unit",
        "execution": {
            "kind": "pytest",
            "command": ["python", "-m", "pytest", *targets, "-q"],
            "timeout_sec": int(raw_execution.get("timeout_sec") or 120),
            "shell": False,
            "env": _merge_env({"PYTHONPATH": "src"}, raw_execution),
        },
        "oracle": {
            "oracle_type": "unit_assertions",
            "required_properties": _oracle_properties(
                raw_proof,
                ["no_test_tampering"],
            ),
            "negative_cases_required": bool(
                (raw_proof.get("oracle") or {}).get("negative_cases_required")
            )
            if isinstance(raw_proof.get("oracle"), dict)
            else False,
        },
        "scope": _template_scope(raw_proof, subtask),
        "anti_gaming": {
            "allows_mock": False,
            "allows_snapshot_update": False,
            "allows_test_only_change": False,
            "requires_real_runtime": False,
        },
        "harness_profile": "python_src_layout",
        "required_capabilities": ["python", "subprocess"],
    }
    return proof


def _desktop_gui_headless_model(
    raw_proof: dict[str, Any], subtask: dict[str, Any]
) -> dict[str, Any]:
    proof = _python_src_unit(raw_proof, subtask)
    proof["template_id"] = "desktop_gui_headless_model"
    proof["harness_profile"] = "desktop_gui_headless"
    proof["oracle"] = {
        **dict(proof["oracle"]),
        "oracle_type": "unit_assertions",
        "required_properties": _oracle_properties(raw_proof, ["no_test_tampering"]),
    }
    return proof


def _desktop_gui_runtime_smoke(
    raw_proof: dict[str, Any], subtask: dict[str, Any]
) -> dict[str, Any]:
    raw_execution = dict(raw_proof.get("execution") or {})
    raw_options = raw_proof.get("harness_options")
    options = dict(raw_options) if isinstance(raw_options, dict) else {}
    package = _infer_package(raw_proof, subtask)
    options.setdefault("managed_runtime", True)
    options.setdefault("readiness", {"type": "process_alive"})
    options.setdefault("startup_timeout_sec", 30)
    options.setdefault("cleanup", {"type": "process_kill"})
    proof = {
        **_base_passthrough(raw_proof),
        "template_id": "desktop_gui_runtime_smoke",
        "execution": {
            "kind": "command",
            "command": ["python", "-m", package],
            "timeout_sec": int(raw_execution.get("timeout_sec") or 180),
            "shell": False,
            "env": _merge_env({"PYTHONPATH": "src"}, raw_execution),
        },
        "oracle": {
            "oracle_type": "behavioral_gui",
            "required_properties": _oracle_properties(
                raw_proof,
                ["runtime_started", "no_test_tampering"],
            ),
            "negative_cases_required": False,
        },
        "scope": {
            "files_under_test": _production_paths(subtask, _proof_scope(raw_proof)),
            "changed_files_expected": _runtime_changed_paths(
                raw_proof,
                subtask,
                _proof_scope(raw_proof),
            ),
            "pytest_targets": _pytest_targets(subtask, _proof_scope(raw_proof)),
        },
        "anti_gaming": {
            "allows_mock": False,
            "allows_snapshot_update": False,
            "allows_test_only_change": False,
            "requires_real_runtime": True,
        },
        "harness_profile": "desktop_gui_runtime",
        "harness_options": options,
        "required_capabilities": ["python", "subprocess", "desktop_gui_runtime"],
    }
    return proof


def instantiate_proof_template(
    proof_payload: dict[str, Any],
    *,
    subtask_payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(proof_payload, dict):
        return proof_payload
    template_id = canonical_proof_template_id(str(proof_payload.get("template_id") or ""))
    if not template_id:
        return proof_payload
    proof_payload = {**proof_payload, "template_id": template_id}
    if template_id == "python_src_unit":
        return _python_src_unit(proof_payload, subtask_payload)
    if template_id == "desktop_gui_headless_model":
        return _desktop_gui_headless_model(proof_payload, subtask_payload)
    if template_id == "desktop_gui_runtime_smoke":
        return _desktop_gui_runtime_smoke(proof_payload, subtask_payload)
    return dict(proof_payload)
