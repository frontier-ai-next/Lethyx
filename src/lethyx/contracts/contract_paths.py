"""Canonical contract paths for recovery and plan-revision deltas.

LLM/watchers may propose typed-looking objects, but path authority belongs to
the PlanIR/ProofContract schema.  This registry is the compiler boundary: any
machine-control delta must normalize here before it can route a phase or block
plan validation.
"""

import ast

from dataclasses import dataclass
from typing import Any, Literal


CANONICAL_CONTRACT_PATHS = frozenset(
    {
        "proof",
        "proof.execution",
        "proof.execution.kind",
        "proof.execution.command",
        "proof.execution.timeout_sec",
        "proof.execution.env",
        "proof.execution.execution_environment_id",
        "proof.template_id",
        "proof.scope",
        "proof.scope.pytest_targets",
        "proof.scope.test_targets",
        "proof.scope.files_under_test",
        "proof.scope.changed_files_expected",
        "proof.oracle",
        "proof.oracle.required_properties",
        "proof.oracle.oracle_type",
        "proof.harness_profile",
        "proof.harness_options",
        "proof.required_capabilities",
        "proof.generated_test_contract",
        "proof.generated_test_contract.oracle_claims",
        "proof.generated_test_contract.proof_budget",
        "proof.anti_gaming.allows_test_only_change",
        "proof.anti_gaming.allows_mock",
        "proof.anti_gaming.allows_snapshot_update",
        "proof.anti_gaming.requires_real_runtime",
        "plan",
        "plan.subtasks",
        "subtask",
        "subtask.dependencies",
        "subtask.files_to_change",
        "subtask.files_to_create",
        "subtask.files_affected",
        "subtask.acceptance_claims",
        "subtask.proof",
        "generated_test_contract",
        "generated_test_contract.oracle_claims",
        "files_to_change",
        "files_to_create",
        "files_affected",
        "acceptance_claims",
    }
)

CONTRACT_PATH_ALIASES = {
    "pytest_targets": "proof.scope.pytest_targets",
    "test_targets": "proof.scope.test_targets",
    "files_under_test": "proof.scope.files_under_test",
    "changed_files_expected": "proof.scope.changed_files_expected",
    "required_properties": "proof.oracle.required_properties",
    "proof.pytest_targets": "proof.scope.pytest_targets",
    "proof.test_targets": "proof.scope.test_targets",
    "proof.files_under_test": "proof.scope.files_under_test",
    "proof.changed_files_expected": "proof.scope.changed_files_expected",
    "proof.required_properties": "proof.oracle.required_properties",
    "proof.oracle.properties": "proof.oracle.required_properties",
    "proof.kind": "proof.execution.kind",
    "proof.command": "proof.execution.command",
    "proof.timeout_sec": "proof.execution.timeout_sec",
    "proof.canonical_proof_template": "proof.template_id",
    "canonical_proof_template": "proof.template_id",
    "template_id": "proof.template_id",
    "harness_profile": "proof.harness_profile",
    "harness_options": "proof.harness_options",
    "required_capabilities": "proof.required_capabilities",
    "dependencies": "subtask.dependencies",
    "subtasks": "plan.subtasks",
    "subtask.dependencies": "subtask.dependencies",
    "subtask.files": "subtask.files_affected",
    "subtask.files_under_test": "proof.scope.files_under_test",
    "subtask.changed_files_expected": "proof.scope.changed_files_expected",
}

ContractPathShape = Literal["scalar", "list", "object"]

SCALAR_CONTRACT_PATHS = frozenset(
    {
        "proof.execution.kind",
        "proof.execution.timeout_sec",
        "proof.execution.execution_environment_id",
        "proof.template_id",
        "proof.oracle.oracle_type",
        "proof.harness_profile",
        "proof.anti_gaming.allows_test_only_change",
        "proof.anti_gaming.allows_mock",
        "proof.anti_gaming.allows_snapshot_update",
        "proof.anti_gaming.requires_real_runtime",
    }
)

LIST_CONTRACT_PATHS = frozenset(
    {
        "proof.execution.command",
        "proof.scope.pytest_targets",
        "proof.scope.test_targets",
        "proof.scope.files_under_test",
        "proof.scope.changed_files_expected",
        "proof.oracle.required_properties",
        "proof.required_capabilities",
        "proof.generated_test_contract.oracle_claims",
        "plan.subtasks",
        "subtask.dependencies",
        "subtask.files_to_change",
        "subtask.files_to_create",
        "subtask.files_affected",
        "subtask.acceptance_claims",
        "generated_test_contract.oracle_claims",
        "files_to_change",
        "files_to_create",
        "files_affected",
        "acceptance_claims",
    }
)

FILE_LIST_CONTRACT_PATHS = frozenset(
    {
        "subtask.files_to_change",
        "subtask.files_to_create",
        "subtask.files_affected",
        "files_to_change",
        "files_to_create",
        "files_affected",
    }
)

OBJECT_CONTRACT_PATHS = frozenset(
    {
        "proof",
        "proof.execution",
        "proof.execution.env",
        "proof.scope",
        "proof.oracle",
        "proof.harness_options",
        "proof.generated_test_contract",
        "proof.generated_test_contract.proof_budget",
        "plan",
        "subtask",
        "subtask.proof",
        "generated_test_contract",
    }
)


@dataclass(frozen=True)
class CanonicalPath:
    path: str

    def __str__(self) -> str:
        return self.path


@dataclass(frozen=True)
class InvalidContractPath(Exception):
    raw_path: str
    message: str
    suggestion: str | None = None
    code: str = "invalid_contract_path"

    def __str__(self) -> str:
        suffix = f"; suggested canonical path: {self.suggestion}" if self.suggestion else ""
        return f"{self.message}: {self.raw_path}{suffix}"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "path": self.raw_path,
            "message": str(self),
        }
        if self.suggestion:
            payload["suggestion"] = self.suggestion
        return payload


@dataclass(frozen=True)
class InvalidContractShape(Exception):
    path: str
    op: str
    reason: str
    required_next_action: str
    code: str = "invalid_contract_shape"

    def __str__(self) -> str:
        return (
            f"{self.code}: {self.reason} "
            f"(path={self.path}, op={self.op}); "
            f"required_next_action={self.required_next_action}"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "path": self.path,
            "op": self.op,
            "reason": self.reason,
            "required_next_action": self.required_next_action,
        }


@dataclass(frozen=True)
class PlanObligationDrop:
    """Reason a generated plan obligation was dropped at the compiler boundary."""

    reason: str
    change_id: str = ""
    path: str = ""
    op: str = ""


def _clean_path(path: str) -> str:
    text = str(path or "").strip().strip("`'\"")
    while text.startswith("."):
        text = text[1:]
    return text.strip()


def _strip_index_segments(path: str) -> str:
    """Drop bracketed index segments for suggestion purposes only."""

    text = _clean_path(path)
    result: list[str] = []
    depth = 0
    for char in text:
        if char == "[":
            depth += 1
            continue
        if char == "]" and depth:
            depth -= 1
            continue
        if depth:
            continue
        result.append(char)
    return "".join(result)


def _subtask_qualified_relative_path(path: str) -> str:
    """Map selected-subtask paths to the relative contract schema path.

    Recovery actions name the target subtask separately. If a model includes the
    subtask id inside the path anyway, treat that as a structured selector prefix,
    not as part of the ProofContract path.
    """

    parts = [part for part in _clean_path(path).split(".") if part]
    if len(parts) >= 3 and parts[0] == "subtasks":
        return ".".join(parts[2:])
    if len(parts) >= 4 and parts[0] == "plan" and parts[1] == "subtasks":
        return ".".join(parts[3:])
    return ""


def _has_forbidden_path_syntax(path: str) -> bool:
    text = _clean_path(path)
    if not text:
        return False
    if any(char in text for char in "[]$*"):
        return True
    if ".." in text or "/" in text or "\\" in text:
        return True
    return False


def suggest_contract_path(path: str) -> str | None:
    text = _strip_index_segments(path)
    if not text:
        return None
    relative = _subtask_qualified_relative_path(text)
    if relative:
        if relative in CONTRACT_PATH_ALIASES:
            return CONTRACT_PATH_ALIASES[relative]
        if relative in CANONICAL_CONTRACT_PATHS:
            return relative
    if text in CONTRACT_PATH_ALIASES:
        return CONTRACT_PATH_ALIASES[text]
    if text in CANONICAL_CONTRACT_PATHS:
        return text
    parts = [part for part in text.split(".") if part]
    while len(parts) > 1:
        parts.pop()
        candidate = ".".join(parts)
        if candidate in CONTRACT_PATH_ALIASES:
            return CONTRACT_PATH_ALIASES[candidate]
        if candidate in CANONICAL_CONTRACT_PATHS:
            return candidate
    return None


def normalize_contract_path(path: str) -> CanonicalPath:
    text = _clean_path(path)
    if not text:
        raise InvalidContractPath(text, "contract path is required")
    if _has_forbidden_path_syntax(text):
        raise InvalidContractPath(
            text,
            "contract path uses unsupported indexed/json/path syntax",
            suggestion=suggest_contract_path(text),
        )
    canonical = CONTRACT_PATH_ALIASES.get(text, text)
    if canonical in CANONICAL_CONTRACT_PATHS:
        return CanonicalPath(canonical)
    relative = _subtask_qualified_relative_path(text)
    if relative:
        canonical = CONTRACT_PATH_ALIASES.get(relative, relative)
        if canonical in CANONICAL_CONTRACT_PATHS:
            return CanonicalPath(canonical)
    raise InvalidContractPath(
        text,
        "unknown contract path",
        suggestion=suggest_contract_path(text),
    )


def is_canonical_contract_path(path: str) -> bool:
    return _clean_path(path) in CANONICAL_CONTRACT_PATHS


def contract_path_shape(path: str) -> ContractPathShape:
    canonical = normalize_contract_path(path).path
    if canonical in SCALAR_CONTRACT_PATHS:
        return "scalar"
    if canonical in LIST_CONTRACT_PATHS:
        return "list"
    if canonical in OBJECT_CONTRACT_PATHS:
        return "object"
    return "object"


def _shape_label(path: str, shape: ContractPathShape) -> str:
    leaf = path.rsplit(".", 1)[-1] if path else "field"
    if shape == "list":
        return f"{leaf} is a typed list"
    if shape == "object":
        return f"{leaf} is a typed object"
    return f"{leaf} is a scalar field"


def _shape_error(
    *,
    code: str,
    path: str,
    op: str,
    reason: str,
    required_next_action: str,
) -> InvalidContractShape:
    return InvalidContractShape(
        path=path,
        op=op,
        reason=reason,
        required_next_action=required_next_action,
        code=code,
    )


def _normalise_required_change_op(op: Any) -> str:
    text = str(op or "").strip().lower()
    if text in {"remove", "delete"}:
        return "remove_applied"
    if text == "replace":
        return "replace_applied"
    return text


def _nearest_semantic_diff_object_path(path: str) -> str:
    parts = [part for part in str(path or "").split(".") if part]
    while parts:
        candidate = ".".join(parts)
        if candidate in OBJECT_CONTRACT_PATHS:
            return candidate
        parts.pop()
    return ""


def _is_scalar_value(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _delta_payload_value(delta: dict[str, Any]) -> Any:
    if "replacement" in delta:
        return delta.get("replacement")
    if "value" in delta:
        return delta.get("value")
    values = delta.get("values")
    if isinstance(values, (list, tuple)):
        return list(values)
    return None


def _payload_items(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "(")):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return [value]
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        return [value]
    if value in (None, ""):
        return []
    return [value]


def _required_change_payload_items(change: dict[str, Any]) -> list[Any]:
    if "values" in change:
        return _payload_items(change.get("values"))
    items: list[Any] = []
    for key in ("value", "new_value", "replacement"):
        if key in change:
            items.extend(_payload_items(change.get(key)))
    return items


def _contract_path_payload_values(change: dict[str, Any]) -> list[str]:
    values = _required_change_payload_items(change)
    if not values:
        return []
    paths: list[str] = []
    for value in values:
        if not isinstance(value, str):
            return []
        try:
            paths.append(normalize_contract_path(value).path)
        except InvalidContractPath:
            return []
    return paths


def _common_contract_path(paths: list[str]) -> str:
    if not paths:
        return ""
    if len(paths) == 1:
        return paths[0]
    split_paths = [path.split(".") for path in paths]
    common: list[str] = []
    for parts in zip(*split_paths):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    while common:
        candidate = ".".join(common)
        try:
            return normalize_contract_path(candidate).path
        except InvalidContractPath:
            common.pop()
    return "subtask"


def _repair_contract_paths_in_file_list_change(
    change: dict[str, Any],
    *,
    path: str,
    op: str,
) -> dict[str, Any]:
    canonical_op = _normalise_required_change_op(op)
    if path not in FILE_LIST_CONTRACT_PATHS or canonical_op not in {"add", "contains"}:
        return change
    contract_paths = _contract_path_payload_values(change)
    if not contract_paths:
        return change
    repaired = dict(change)
    repaired["path"] = _common_contract_path(contract_paths)
    repaired["op"] = "semantic_diff"
    for key in ("value", "values", "new_value", "old_value", "replacement"):
        repaired.pop(key, None)
    return repaired


def _coarsen_exact_proof_command_pin(
    change: dict[str, Any],
    *,
    path: str,
    op: str,
) -> tuple[dict[str, Any], str, str]:
    canonical_op = _normalise_required_change_op(op)
    if path != "proof.execution.command":
        return change, path, op
    if canonical_op in {"exists", "semantic_diff"}:
        return change, path, op
    repaired = dict(change)
    repaired["path"] = "proof.execution"
    repaired["op"] = "semantic_diff"
    for key in ("value", "values", "new_value", "old_value", "replacement"):
        repaired.pop(key, None)
    return repaired, "proof.execution", "semantic_diff"


def _coarsen_agent_review_plan_subtasks_pin(
    change: dict[str, Any],
    *,
    path: str,
    op: str,
    source: str,
) -> tuple[dict[str, Any], str, str]:
    if source != "agent_review" or path != "plan.subtasks":
        return change, path, op
    canonical_op = _normalise_required_change_op(op)
    if canonical_op not in {"add", "contains", "replace_applied"}:
        return change, path, op
    payload_items = _required_change_payload_items(change)
    if not payload_items or not any(
        isinstance(item, (dict, list, tuple)) for item in payload_items
    ):
        return change, path, op
    repaired = dict(change)
    repaired["path"] = "plan.subtasks"
    repaired["op"] = "semantic_diff"
    for key in ("value", "values", "new_value", "old_value", "replacement"):
        repaired.pop(key, None)
    return repaired, "plan.subtasks", "semantic_diff"


def _coarsen_object_path_value_pin(
    change: dict[str, Any],
    *,
    path: str,
    op: str,
) -> tuple[dict[str, Any], str, str]:
    """Coarsen an unsatisfiable object-field obligation to a ``semantic_diff``.

    ``add``/``contains``/``equals`` on an object contract path (e.g.
    ``proof.harness_options``) is only well-defined for a dict-subset payload.
    When a reviewer pins an object sub-field (``proof.harness_options.assert_command``
    collapses onto ``proof.harness_options``) or hands a scalar/list payload such
    as an argv, a token-wise ``contains`` check can never be satisfied and the
    plan loops into an impasse. Those changes express a structured object
    revision, so enforce them as a ``semantic_diff`` the agent applies through
    apply_plan_revision_patch instead.
    """

    raw = _clean_path(path)
    if not raw:
        return change, path, op
    try:
        canonical = normalize_contract_path(raw).path
    except InvalidContractPath as exc:
        canonical = ""
        if exc.suggestion:
            try:
                canonical = normalize_contract_path(exc.suggestion).path
            except InvalidContractPath:
                canonical = ""
    if not canonical or canonical not in OBJECT_CONTRACT_PATHS:
        return change, path, op
    is_collapsed_subfield = raw != canonical and raw.startswith(canonical + ".")
    canonical_op = _normalise_required_change_op(op)
    if canonical_op == "add":
        canonical_op = "contains"
    if canonical_op in {"exists", "semantic_diff", "remove_applied", "not_contains"}:
        if not is_collapsed_subfield:
            return change, path, op
    elif canonical_op not in {"contains", "equals", "replace_applied"}:
        return change, path, op
    payload_items = _required_change_payload_items(change)
    payload_non_dict = bool(payload_items) and any(
        not isinstance(item, dict) for item in payload_items
    )
    if not (is_collapsed_subfield or payload_non_dict):
        return change, path, op
    repaired = dict(change)
    repaired["path"] = canonical
    repaired["op"] = "semantic_diff"
    for key in ("value", "values", "new_value", "old_value", "replacement"):
        repaired.pop(key, None)
    return repaired, canonical, "semantic_diff"


def _coarsen_agent_review_value_less_list_remove(
    change: dict[str, Any],
    *,
    path: str,
    op: str,
    source: str,
) -> tuple[dict[str, Any], str, str]:
    """Route value-less review removals on list fields through semantic diff.

    Agent reviewers sometimes express a concrete list-item removal in the
    typed ``required_plan_changes`` object as ``op=remove`` on a list path while
    omitting the removed value. That object is still typed enough to route to a
    subtask/path, but it is not exact enough for an item-level
    ``remove_applied`` check: requiring the whole list field to disappear makes
    the subsequent plan revision impossible to satisfy. Treat that shape as a
    structured change to the list and let the review-normalization boundary add
    the previous list value for semantic-diff enforcement.
    """

    if source != "agent_review":
        return change, path, op
    canonical_op = _normalise_required_change_op(op)
    if canonical_op != "remove_applied":
        return change, path, op
    if _required_change_payload_items(change):
        return change, path, op
    try:
        shape = contract_path_shape(path)
    except InvalidContractPath:
        return change, path, op
    if shape != "list":
        return change, path, op
    repaired = dict(change)
    repaired["op"] = "semantic_diff"
    return repaired, path, "semantic_diff"


def validate_required_change_shape(
    change: dict[str, Any],
    schema: Any = None,
) -> dict[str, Any]:
    """Validate operation shape against the canonical contract field type.

    The optional ``schema`` argument is reserved for future schema-specific
    registries; the current boundary is the canonical PlanIR/ProofSpec path
    registry in this module.
    """

    del schema
    if not isinstance(change, dict):
        raise _shape_error(
            code="invalid_required_change_shape",
            path="",
            op="",
            reason="required_plan_change must be a typed object",
            required_next_action=(
                "produce a typed required_plan_change object or mark recovery_contract_invalid"
            ),
        )
    normalized = dict(change)
    path = normalize_contract_path(
        str(normalized.get("path") or normalized.get("field") or "")
    ).path
    normalized["path"] = path
    raw_op = str(normalized.get("op") or normalized.get("action") or "").strip().lower()
    op = _normalise_required_change_op(raw_op)
    shape = contract_path_shape(path)
    if raw_op == "add":
        if shape in {"list", "object"}:
            op = "contains"
        else:
            raise _shape_error(
                code="invalid_required_change_shape",
                path=path,
                op=raw_op,
                reason=f"{_shape_label(path, shape)} cannot use add",
                required_next_action=(
                    "use equals for scalar fields or a list/object contract path"
                ),
            )
    elif raw_op == "replace" and shape == "object":
        op = "semantic_diff"
    normalized["op"] = op
    if op not in {
        "exists",
        "equals",
        "contains",
        "not_contains",
        "remove_applied",
        "replace_applied",
        "semantic_diff",
    }:
        raise _shape_error(
            code="invalid_required_change_shape",
            path=path,
            op=op,
            reason=f"required_plan_change op `{op}` is not supported",
            required_next_action=(
                "use exists, equals, contains, not_contains, remove_applied, "
                "replace_applied, or semantic_diff"
            ),
        )
    if op == "equals" and shape != "scalar":
        label = _shape_label(path, shape)
        prose = "prose equality is not a valid contract operation"
        raise _shape_error(
            code="invalid_required_change_shape",
            path=path,
            op=op,
            reason=f"{label}; {prose}",
            required_next_action=(
                "produce ContractDelta remove/replace or mark recovery_contract_invalid"
            ),
        )
    if op == "equals" and not _is_scalar_value(_delta_payload_value(normalized)):
        raise _shape_error(
            code="invalid_required_change_shape",
            path=path,
            op=op,
            reason=f"{_shape_label(path, shape)} cannot equal a structured value",
            required_next_action=(
                "use semantic_diff for structured plan revisions or produce "
                "a scalar equality change"
            ),
        )
    return normalized


def _plan_obligation_id(change: dict[str, Any]) -> str:
    return str(change.get("id") or change.get("change_id") or "").strip()


def _record_obligation_drop(
    drop_reasons: list[PlanObligationDrop] | None,
    *,
    reason: str,
    change: dict[str, Any],
    path: str = "",
    op: str = "",
) -> None:
    if drop_reasons is None:
        return
    drop_reasons.append(
        PlanObligationDrop(
            reason=reason,
            change_id=_plan_obligation_id(change),
            path=path or str(change.get("path") or change.get("field") or ""),
            op=op or str(change.get("op") or change.get("action") or ""),
        )
    )


def _coarsened_semantic_diff_change(
    change: dict[str, Any],
    *,
    path: str,
) -> dict[str, Any] | None:
    try:
        if contract_path_shape(path) == "scalar":
            return None
        return validate_required_change_shape({**change, "path": path, "op": "semantic_diff"})
    except (InvalidContractPath, InvalidContractShape):
        return None


def _normalize_obligation_path(item: dict[str, Any]) -> str:
    raw_path = str(item.get("path") or item.get("field") or "").strip()
    try:
        return normalize_contract_path(raw_path).path
    except InvalidContractPath as exc:
        if exc.suggestion:
            return normalize_contract_path(exc.suggestion).path
        raise


def normalize_plan_obligation(
    change: dict[str, Any],
    *,
    plan: dict[str, Any] | None = None,
    obligation_source: str | None = "control_plane",
    drop_reasons: list[PlanObligationDrop] | None = None,
) -> dict[str, Any] | None:
    """Normalize a required_plan_change at its generation/enforcement boundary.

    This is intentionally stricter than ad hoc callers and more forgiving than
    raw shape validation: server-authored obligations are repaired or coarsened
    before they reach the plan validator. Only genuinely unroutable changes are
    dropped, with an optional machine-readable drop reason for tests/logging.
    """

    del plan  # reserved for future schema-aware routing.
    source = str(obligation_source or "").strip()
    server_authored = source == "control_plane"
    if not isinstance(change, dict):
        _record_obligation_drop(
            drop_reasons,
            reason="not_a_typed_object",
            change={},
        )
        return None
    item = dict(change)
    path = ""
    op = str(item.get("op") or item.get("action") or "").strip().lower()
    if not op:
        _record_obligation_drop(
            drop_reasons,
            reason="missing_op",
            change=item,
        )
        return None
    if op == "replace" and "new_value" not in item and "value" in item:
        item["new_value"] = item.get("value")
    try:
        path = _normalize_obligation_path(item)
    except InvalidContractPath:
        _record_obligation_drop(
            drop_reasons,
            reason="invalid_path",
            change=item,
            op=op,
        )
        return None
    item = _repair_contract_paths_in_file_list_change(item, path=path, op=op)
    path = str(item.get("path") or path)
    op = str(item.get("op") or op).strip().lower()
    item, path, op = _coarsen_exact_proof_command_pin(item, path=path, op=op)
    item, path, op = _coarsen_agent_review_plan_subtasks_pin(
        item,
        path=path,
        op=op,
        source=source,
    )
    item, path, op = _coarsen_object_path_value_pin(item, path=path, op=op)
    item, path, op = _coarsen_agent_review_value_less_list_remove(
        item,
        path=path,
        op=op,
        source=source,
    )
    if (
        source == "agent_review"
        and op == "replace_applied"
        and "old_value" not in item
        and "new_value" not in item
    ):
        semantic_path = _nearest_semantic_diff_object_path(path)
        if semantic_path:
            item["path"] = semantic_path
            item["op"] = "semantic_diff"
            for key in ("value", "values", "replacement"):
                item.pop(key, None)
            path = semantic_path
            op = "semantic_diff"
    try:
        normalized = validate_required_change_shape({**item, "path": path, "op": op})
    except InvalidContractShape:
        if not server_authored:
            raise
        normalized = _coarsened_semantic_diff_change(item, path=path)
        if normalized is None:
            _record_obligation_drop(
                drop_reasons,
                reason="invalid_shape",
                change=item,
                path=path,
                op=op,
            )
            return None
    except InvalidContractPath:
        _record_obligation_drop(
            drop_reasons,
            reason="invalid_path",
            change=item,
            path=path,
            op=op,
        )
        return None

    canonical_path = str(normalized.get("path") or path)
    if source:
        normalized["obligation_source"] = str(normalized.get("obligation_source") or source)
    normalized.setdefault("obligation_code", "plan_obligation")
    if canonical_path == "plan" or canonical_path.startswith("plan."):
        for key in ("target_subtask_id", "subtask_id"):
            normalized.pop(key, None)
        normalized["target_kind"] = "plan"
    else:
        normalized.setdefault("target_kind", "subtask")
    return normalized


def validate_contract_delta_shape(
    delta: dict[str, Any],
    schema: Any = None,
) -> dict[str, Any]:
    """Normalize a ContractDelta and reject impossible operations by field type."""

    del schema
    normalized = validate_delta_path(delta)
    op = str(normalized.get("op") or "").strip().lower()
    # Accept the required_plan_change op vocabulary that reviewers/recovery
    # naturally reach for. ContractDelta only has add/remove/replace, but the
    # model frequently emits replace_applied/remove_applied/delete/update/set;
    # coerce them rather than rejecting an otherwise-valid structural delta.
    _DELTA_OP_ALIASES = {
        "replace_applied": "replace",
        "remove_applied": "remove",
        "delete": "remove",
        "del": "remove",
        "update": "replace",
        "set": "replace",
        "insert": "add",
        "append": "add",
    }
    if op in _DELTA_OP_ALIASES:
        op = _DELTA_OP_ALIASES[op]
        normalized["op"] = op
    path = str(normalized.get("path") or "")
    shape = contract_path_shape(path)
    if op == "equals":
        raise _shape_error(
            code="invalid_contract_delta_shape",
            path=path,
            op=op,
            reason="ContractDelta does not support equals; use add/remove/replace",
            required_next_action=(
                "produce ContractDelta remove/replace/add or mark recovery_contract_invalid"
            ),
        )
    if op not in {"add", "remove", "replace"}:
        return normalized

    value = _delta_payload_value(normalized)
    if op == "remove":
        if shape == "list" and not isinstance(normalized.get("values"), (list, tuple)):
            raise _shape_error(
                code="invalid_contract_delta_shape",
                path=path,
                op=op,
                reason=f"{_shape_label(path, shape)} removals must name values",
                required_next_action=(
                    "produce ContractDelta remove with values=[...] or mark "
                    "recovery_contract_invalid"
                ),
            )
        return normalized

    if shape == "scalar" and not _is_scalar_value(value):
        raise _shape_error(
            code="invalid_contract_delta_shape",
            path=path,
            op=op,
            reason=f"{_shape_label(path, shape)} requires a scalar value",
            required_next_action=(
                "produce a scalar ContractDelta value or mark recovery_contract_invalid"
            ),
        )
    if shape == "list" and op == "replace" and not isinstance(value, list):
        raise _shape_error(
            code="invalid_contract_delta_shape",
            path=path,
            op=op,
            reason=f"{_shape_label(path, shape)} replacement must be a list",
            required_next_action=(
                "produce ContractDelta remove/add for list members, replace "
                "with a list, or mark recovery_contract_invalid"
            ),
        )
    if shape == "object" and value is not None and not isinstance(value, dict):
        raise _shape_error(
            code="invalid_contract_delta_shape",
            path=path,
            op=op,
            reason=f"{_shape_label(path, shape)} requires an object value",
            required_next_action=(
                "produce a structured ContractDelta value or mark recovery_contract_invalid"
            ),
        )
    return normalized


def validate_delta_path(delta: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(delta, dict):
        raise InvalidContractPath("", "contract delta must be an object")
    normalized = dict(delta)
    canonical = normalize_contract_path(str(normalized.get("path") or ""))
    normalized["path"] = canonical.path
    return normalized


__all__ = [
    "CANONICAL_CONTRACT_PATHS",
    "CONTRACT_PATH_ALIASES",
    "CanonicalPath",
    "ContractPathShape",
    "InvalidContractPath",
    "InvalidContractShape",
    "is_canonical_contract_path",
    "contract_path_shape",
    "normalize_contract_path",
    "suggest_contract_path",
    "validate_contract_delta_shape",
    "validate_delta_path",
    "validate_required_change_shape",
]
