"""Deterministic recovery options derived from typed issues.

Reviewers/watchers diagnose problems.  They do not own PlanIR paths.  This
module maps issue codes plus current control-plane knowledge into canonical
recovery options that the runner can route and validators can enforce.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from lethyx.contracts.contract_paths import (
    InvalidContractPath,
    InvalidContractShape,
    contract_path_shape,
    normalize_contract_path,
    normalize_plan_obligation,
)
from lethyx.contracts.models import ContractDelta, json_ready


BAD_ORACLE_CODES = frozenset(
    {
        "bad_generated_oracle",
        "invalid_generated_test_contract",
        "inconsistent_generated_oracle",
        "oracle_domain_mismatch",
        "contradictory_required_behavior",
        "unmapped_generated_oracle",
    }
)
_BAD_ORACLE_REQUIRED_PROPERTY_PATHS = frozenset(
    {
        "proof.oracle.required_properties",
        "proof.required_properties",
    }
)
_BAD_ORACLE_CLAIMS_PATHS = frozenset(
    {
        "generated_test_contract.oracle_claims",
        "proof.generated_test_contract.oracle_claims",
    }
)
_KNOWN_UNGROUNDED_PROPERTY_IDS = frozenset(
    {
        "distinct_inputs_distinct_outputs",
    }
)


@dataclass(frozen=True)
class RecoveryOption:
    code: str
    target_subtask_id: str
    reason_code: str
    required_plan_changes: tuple[dict[str, Any], ...] = ()
    required_deltas: tuple[ContractDelta, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized: list[dict[str, Any]] = []
        for change in self.required_plan_changes:
            item = normalize_plan_obligation(
                dict(change),
                obligation_source="control_plane",
            )
            if item is not None:
                normalized.append(item)
        object.__setattr__(self, "required_plan_changes", tuple(normalized))

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "target_subtask_id": self.target_subtask_id,
            "reason_code": self.reason_code,
        }
        if self.required_plan_changes:
            payload["required_plan_changes"] = [
                json_ready(item) for item in self.required_plan_changes
            ]
        if self.required_deltas:
            payload["required_deltas"] = [item.to_payload() for item in self.required_deltas]
        if self.evidence_refs:
            payload["evidence_refs"] = list(self.evidence_refs)
        return payload


def _evidence_refs(issue: Mapping[str, Any]) -> tuple[str, ...]:
    raw = issue.get("evidence_refs")
    return (
        tuple(str(item).strip() for item in (raw or []) if str(item).strip())
        if isinstance(raw, list)
        else ()
    )


def _target_subtask_id(issue: Mapping[str, Any]) -> str:
    return str(issue.get("target_subtask_id") or issue.get("subtask_id") or "").strip()


def _canonical_path(raw: Any) -> str:
    try:
        return normalize_contract_path(str(raw or "")).path
    except (InvalidContractPath, InvalidContractShape):
        return ""


def _invalid_values(issue: Mapping[str, Any]) -> tuple[str, ...]:
    raw = issue.get("invalid_values")
    if not isinstance(raw, list):
        raw = issue.get("invalid_required_properties")
    return (
        tuple(str(item).strip() for item in (raw or []) if str(item).strip())
        if isinstance(raw, list)
        else ()
    )


def _canonical_deltas(issue: Mapping[str, Any]) -> tuple[ContractDelta, ...]:
    raw = issue.get("required_deltas")
    if not isinstance(raw, list):
        return ()
    deltas: list[ContractDelta] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            deltas.append(ContractDelta.from_mapping(item))
        except (InvalidContractPath, InvalidContractShape, TypeError, ValueError):
            continue
    return tuple(deltas)


def _required_change_op_for_delta(delta: ContractDelta) -> str:
    if delta.op == "remove":
        return "remove_applied"
    if delta.op == "add":
        return "contains"
    if delta.op == "replace" and contract_path_shape(delta.path) == "scalar":
        return "equals"
    return "semantic_diff"


def _delta_values(delta: ContractDelta) -> list[Any]:
    if delta.values:
        return list(delta.values)
    if delta.value is not None:
        return [delta.value]
    if delta.replacement is not None:
        return [delta.replacement]
    return [None]


def _plan_changes_from_deltas(
    *,
    deltas: tuple[ContractDelta, ...],
    target_subtask_id: str,
    reason_code: str,
    source: str,
    evidence_refs: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    changes: list[dict[str, Any]] = []
    for index, delta in enumerate(deltas, start=1):
        op = _required_change_op_for_delta(delta)
        values = _delta_values(delta)
        if op == "semantic_diff":
            normalized = normalize_plan_obligation(
                {
                    "id": f"{reason_code}-{index}",
                    "target_subtask_id": target_subtask_id,
                    "severity": "blocking",
                    "reason_code": reason_code,
                    "source": source,
                    "path": delta.path,
                    "op": "semantic_diff",
                    "evidence_refs": list(evidence_refs),
                },
                obligation_source="control_plane",
            )
            if normalized is not None:
                changes.append(normalized)
            continue
        for value_index, value in enumerate(values, start=1):
            change = {
                "id": f"{reason_code}-{index}-{value_index}",
                "target_subtask_id": target_subtask_id,
                "severity": "blocking",
                "reason_code": reason_code,
                "source": source,
                "path": delta.path,
                "op": op,
                "evidence_refs": list(evidence_refs),
            }
            if value is not None:
                change["value"] = json_ready(value)
            normalized = normalize_plan_obligation(
                change,
                obligation_source="control_plane",
            )
            if normalized is not None:
                changes.append(normalized)
    return tuple(changes)


def _capability_env_deltas_are_noop(
    *,
    issue: Mapping[str, Any],
    deltas: tuple[ContractDelta, ...],
) -> bool:
    if not deltas:
        return False
    env_id = str(issue.get("env_id") or "").strip()
    if not env_id:
        return False
    for delta in deltas:
        if delta.op != "replace":
            return False
        if delta.path != "proof.execution.execution_environment_id":
            return False
        if str(delta.replacement or delta.value or "").strip() != env_id:
            return False
    return True


def _bad_oracle_recovery_options(issue: Mapping[str, Any]) -> tuple[RecoveryOption, ...]:
    code = str(issue.get("code") or "").strip()
    target_subtask_id = _target_subtask_id(issue)
    evidence_refs = _evidence_refs(issue)
    source = f"RecoveryPolicy.{code}"
    deltas = _canonical_deltas(issue)
    if deltas:
        return (
            RecoveryOption(
                code="oracle_repair",
                target_subtask_id=target_subtask_id,
                reason_code=code,
                required_plan_changes=_plan_changes_from_deltas(
                    deltas=deltas,
                    target_subtask_id=target_subtask_id,
                    reason_code=code,
                    source=source,
                    evidence_refs=evidence_refs,
                ),
                required_deltas=deltas,
                evidence_refs=evidence_refs,
            ),
        )

    invalid_values = _invalid_values(issue)
    contract_path = _canonical_path(issue.get("contract_path"))
    if contract_path in _BAD_ORACLE_REQUIRED_PROPERTY_PATHS and invalid_values:
        path = "proof.oracle.required_properties"
        delta = ContractDelta(
            op="remove",
            path=path,
            values=invalid_values,
            target_subtask_id=target_subtask_id,
            source_issue_code=code,
        )
        return (
            RecoveryOption(
                code="oracle_repair",
                target_subtask_id=target_subtask_id,
                reason_code=code,
                required_plan_changes=_plan_changes_from_deltas(
                    deltas=(delta,),
                    target_subtask_id=target_subtask_id,
                    reason_code=code,
                    source=source,
                    evidence_refs=evidence_refs,
                ),
                required_deltas=(delta,),
                evidence_refs=evidence_refs,
            ),
        )
    if contract_path in _BAD_ORACLE_CLAIMS_PATHS and invalid_values:
        delta = ContractDelta(
            op="remove",
            path=contract_path,
            values=invalid_values,
            target_subtask_id=target_subtask_id,
            source_issue_code=code,
        )
        return (
            RecoveryOption(
                code="oracle_repair",
                target_subtask_id=target_subtask_id,
                reason_code=code,
                required_plan_changes=_plan_changes_from_deltas(
                    deltas=(delta,),
                    target_subtask_id=target_subtask_id,
                    reason_code=code,
                    source=source,
                    evidence_refs=evidence_refs,
                ),
                required_deltas=(delta,),
                evidence_refs=evidence_refs,
            ),
        )
    known_properties = tuple(
        value for value in invalid_values if value in _KNOWN_UNGROUNDED_PROPERTY_IDS
    )
    if known_properties:
        delta = ContractDelta(
            op="remove",
            path="proof.oracle.required_properties",
            values=known_properties,
            target_subtask_id=target_subtask_id,
            source_issue_code=code,
        )
        return (
            RecoveryOption(
                code="oracle_repair",
                target_subtask_id=target_subtask_id,
                reason_code=code,
                required_plan_changes=_plan_changes_from_deltas(
                    deltas=(delta,),
                    target_subtask_id=target_subtask_id,
                    reason_code=code,
                    source=source,
                    evidence_refs=evidence_refs,
                ),
                required_deltas=(delta,),
                evidence_refs=evidence_refs,
            ),
        )
    return (
        RecoveryOption(
            code="recovery_contract_invalid",
            target_subtask_id=target_subtask_id,
            reason_code=code or "bad_oracle_without_typed_delta",
            evidence_refs=evidence_refs,
        ),
    )


def derive_recovery_options(
    issue: Mapping[str, Any],
    *,
    runtime_capability_available: bool = False,
) -> tuple[RecoveryOption, ...]:
    code = str(issue.get("code") or "").strip()
    target_subtask_id = _target_subtask_id(issue)
    evidence_refs = _evidence_refs(issue)
    if code in BAD_ORACLE_CODES:
        return _bad_oracle_recovery_options(issue)
    if code == "proof_execution_env_mismatch":
        source = f"RecoveryPolicy.{code}"
        return (
            RecoveryOption(
                code="proof_contract_repair",
                target_subtask_id=target_subtask_id,
                reason_code=code,
                required_plan_changes=(
                    {
                        "id": "proof-runtime-env-alignment",
                        "target_subtask_id": target_subtask_id,
                        "severity": "blocking",
                        "reason_code": code,
                        "source": source,
                        "path": "proof.execution.env",
                        "op": "semantic_diff",
                        "evidence_refs": list(evidence_refs),
                    },
                ),
                evidence_refs=evidence_refs,
            ),
            RecoveryOption(
                code="plan_contract_revision",
                target_subtask_id=target_subtask_id,
                reason_code=code,
                required_plan_changes=(
                    {
                        "id": "proof-runtime-contract-alignment",
                        "target_subtask_id": target_subtask_id,
                        "severity": "blocking",
                        "reason_code": code,
                        "source": source,
                        "path": "proof.execution",
                        "op": "semantic_diff",
                        "evidence_refs": list(evidence_refs),
                    },
                ),
                evidence_refs=evidence_refs,
            ),
        )

    if code in {
        "capability_probe_failed",
        "capability_probe_environment_mismatch",
        "dependency_provision_required",
    }:
        source = f"RecoveryPolicy.{code}"
        deltas = _canonical_deltas(issue)
        if code == "capability_probe_environment_mismatch" and _capability_env_deltas_are_noop(
            issue=issue,
            deltas=deltas,
        ):
            capability_id = str(issue.get("capability_id") or "").strip()
            env_id = str(issue.get("env_id") or "").strip()
            change = {
                "id": f"capability-reprobe-{capability_id or code}",
                "target_subtask_id": target_subtask_id,
                "target_phase": "execute",
                "severity": "blocking",
                "reason_code": code,
                "source": source,
                "path": _canonical_path(issue.get("contract_path"))
                or "proof.required_capabilities",
                "op": "semantic_diff",
                "action": "re_probe_capability",
                "evidence_refs": list(evidence_refs),
            }
            if capability_id:
                change["capability_id"] = capability_id
            if env_id:
                change["env_id"] = env_id
            return (
                RecoveryOption(
                    code="re_probe_capability",
                    target_subtask_id=target_subtask_id,
                    reason_code=code,
                    required_plan_changes=(change,),
                    evidence_refs=evidence_refs,
                ),
            )
        if deltas:
            return (
                RecoveryOption(
                    code="proof_contract_repair",
                    target_subtask_id=target_subtask_id,
                    reason_code=code,
                    required_plan_changes=_plan_changes_from_deltas(
                        deltas=deltas,
                        target_subtask_id=target_subtask_id,
                        reason_code=code,
                        source=source,
                        evidence_refs=evidence_refs,
                    ),
                    required_deltas=deltas,
                    evidence_refs=evidence_refs,
                ),
            )
        capability_id = str(issue.get("capability_id") or "").strip()
        env_id = str(issue.get("env_id") or "").strip()
        change = {
            "id": f"capability-repair-{capability_id or code}",
            "target_subtask_id": target_subtask_id,
            "target_phase": "execute",
            "severity": "blocking",
            "reason_code": code,
            "source": source,
            "path": _canonical_path(issue.get("contract_path")) or "proof.required_capabilities",
            "op": "semantic_diff",
            "evidence_refs": list(evidence_refs),
        }
        if capability_id:
            change["capability_id"] = capability_id
        if env_id:
            change["env_id"] = env_id
        return (
            RecoveryOption(
                code="env_capability_repair",
                target_subtask_id=target_subtask_id,
                reason_code=code,
                required_plan_changes=(change,),
                evidence_refs=evidence_refs,
            ),
        )

    if code in {
        "package_import_env_mismatch",
        "setup_harness_mismatch",
    }:
        source = f"RecoveryPolicy.{code}"
        language = (
            str(
                issue.get("language")
                or issue.get("ecosystem")
                or issue.get("project_type")
                or issue.get("stack")
                or "python"
            )
            .strip()
            .lower()
        )
        if language not in {"", "python", "py"}:
            return (
                RecoveryOption(
                    code="proof_contract_repair",
                    target_subtask_id=target_subtask_id,
                    reason_code=code,
                    required_plan_changes=(
                        {
                            "id": f"{language}-proof-runtime-alignment",
                            "target_subtask_id": target_subtask_id,
                            "severity": "blocking",
                            "reason_code": code,
                            "source": source,
                            "path": "proof.execution",
                            "op": "semantic_diff",
                            "evidence_refs": list(evidence_refs),
                        },
                    ),
                    evidence_refs=evidence_refs,
                ),
            )
        return (
            RecoveryOption(
                code="proof_contract_repair",
                target_subtask_id=target_subtask_id,
                reason_code=code,
                required_plan_changes=(
                    {
                        "id": "pytest-src-layout-pythonpath",
                        "target_subtask_id": target_subtask_id,
                        "severity": "blocking",
                        "reason_code": code,
                        "source": source,
                        "path": "proof.execution.env",
                        "op": "contains",
                        "value": {"PYTHONPATH": "src"},
                        "evidence_refs": list(evidence_refs),
                    },
                ),
                required_deltas=(
                    ContractDelta(
                        op="add",
                        path="proof.execution.env",
                        value={"PYTHONPATH": "src"},
                        target_subtask_id=target_subtask_id,
                        source_issue_code=code,
                    ),
                ),
                evidence_refs=evidence_refs,
            ),
            RecoveryOption(
                code="packaging_import_repair",
                target_subtask_id=target_subtask_id,
                reason_code=code,
                required_plan_changes=(
                    {
                        "id": "pytest-src-layout-packaging",
                        "target_subtask_id": target_subtask_id,
                        "severity": "blocking",
                        "reason_code": code,
                        "source": source,
                        "path": "files_to_change",
                        "op": "semantic_diff",
                        "allowed_files": [
                            "pyproject.toml",
                            "pytest.ini",
                            "setup.cfg",
                            "workspace.toml",
                        ],
                        "forbidden_files": ["src/", "tests/"],
                        "evidence_refs": list(evidence_refs),
                    },
                ),
                evidence_refs=evidence_refs,
            ),
            RecoveryOption(
                code="plan_contract_revision",
                target_subtask_id=target_subtask_id,
                reason_code=code,
                required_plan_changes=(
                    {
                        "id": "project-setup-harness-alignment",
                        "target_subtask_id": target_subtask_id,
                        "severity": "blocking",
                        "reason_code": code,
                        "source": source,
                        "path": "proof.execution",
                        "op": "semantic_diff",
                        "evidence_refs": list(evidence_refs),
                    },
                ),
                evidence_refs=evidence_refs,
            ),
        )

    if code != "headless_proof_uses_real_gui_root":
        return ()

    options = [
        RecoveryOption(
            code="replace_with_headless_controller_proof",
            target_subtask_id=target_subtask_id,
            reason_code=code,
            required_plan_changes=(
                {
                    "id": "headless-real-root-proof-target",
                    "target_subtask_id": target_subtask_id,
                    "severity": "blocking",
                    "reason_code": code,
                    "source": "RecoveryPolicy.headless_proof_uses_real_gui_root",
                    "path": "proof.scope.pytest_targets",
                    "op": "semantic_diff",
                    "evidence_refs": list(evidence_refs),
                },
            ),
            evidence_refs=evidence_refs,
        )
    ]
    if runtime_capability_available:
        options.append(
            RecoveryOption(
                code="upgrade_to_desktop_gui_runtime_proof",
                target_subtask_id=target_subtask_id,
                reason_code=code,
                required_deltas=(
                    ContractDelta(
                        op="replace",
                        path="proof.harness_profile",
                        value="desktop_gui_runtime",
                        target_subtask_id=target_subtask_id,
                        source_issue_code=code,
                    ),
                    ContractDelta(
                        op="add",
                        path="proof.required_capabilities",
                        values=("desktop_gui_runtime",),
                        target_subtask_id=target_subtask_id,
                        source_issue_code=code,
                    ),
                ),
                evidence_refs=evidence_refs,
            )
        )
    return tuple(options)


__all__ = ["BAD_ORACLE_CODES", "RecoveryOption", "derive_recovery_options"]
