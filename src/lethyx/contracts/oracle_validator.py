"""Deterministic checks for generated oracle/test contracts.

The LLM may propose generated tests, but conflicting test expectations are a
contract problem, not an implementation problem. This module turns typed
``GeneratedTestContract`` data into machine-readable ContractIssues so the
control plane can route to plan revision without scraping review prose.
"""

import json
import re
from typing import Any

from lethyx.contracts.models import ContractIssue, EvidenceRef

BAD_ORACLE_REVIEW_CODES = frozenset(
    {
        "bad_generated_oracle",
        "plan_contract_issue",
        "inconsistent_generated_oracle",
        "oracle_domain_mismatch",
        "contradictory_required_behavior",
        "invalid_generated_test_contract",
        "unmapped_generated_oracle",
    }
)

_VALID_CLAIM_SOURCES = frozenset(
    {
        "task_requirement",
        "interface_model",
        "reference_behavior",
        "harness_contract",
    }
)
_DEFAULT_MAX_GENERATED_TESTS_PER_SUBTASK = 6


def extract_failed_pytest_node_ids(output: str) -> tuple[str, ...]:
    """Extract pytest node ids from machine-formatted failure summary text."""

    nodes: list[str] = []
    for line in str(output or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("FAILED "):
            continue
        node = stripped[len("FAILED ") :].split(" - ", 1)[0].strip()
        if node and node not in nodes:
            nodes.append(node.replace("\\", "/"))
    return tuple(nodes)


def extract_failed_jest_targets(output: str) -> tuple[str, ...]:
    """Extract failed Jest/Vitest test file targets from common summary lines."""

    targets: list[str] = []
    for line in str(output or "").splitlines():
        stripped = line.strip()
        match = re.search(r"\bFAIL\b\s+([^\s]+)", stripped)
        if not match:
            match = re.search(r"\bFAILED\b\s+([^\s]+)", stripped)
        if match:
            target = match.group(1).replace("\\", "/")
            if target and target not in targets:
                targets.append(target)
    return tuple(targets)


def extract_failed_go_test_targets(output: str) -> tuple[str, ...]:
    """Extract failed Go package/test targets from `go test` output."""

    targets: list[str] = []
    for line in str(output or "").splitlines():
        stripped = line.strip()
        match = re.match(r"--- FAIL:\s+([^\s(]+)", stripped)
        if not match:
            match = re.match(r"FAIL\s+([^\s]+)", stripped)
        if match:
            target = match.group(1).replace("\\", "/")
            if target and target not in targets:
                targets.append(target)
    return tuple(targets)


def extract_failed_cargo_test_targets(output: str) -> tuple[str, ...]:
    """Extract failed Rust test names from `cargo test` output."""

    targets: list[str] = []
    for line in str(output or "").splitlines():
        stripped = line.strip()
        match = re.match(r"test\s+(.+?)\s+\.\.\.\s+FAILED\b", stripped)
        if not match:
            match = re.match(r"failures:\s*(.*)", stripped, flags=re.I)
        if match:
            value = match.group(1).strip()
            if value and value not in targets:
                targets.append(value)
    return tuple(targets)


def generated_oracle_contract_issues(
    contract: Any,
    *,
    subtask_id: str = "",
    failed_node_ids: tuple[str, ...] = (),
    evidence_refs: tuple[str, ...] = (),
) -> list[ContractIssue]:
    """Return blocking issues for contradictory generated oracle claims."""

    if not isinstance(contract, dict) or not contract:
        return []
    depth_limit_paths = _depth_limit_placeholder_paths(contract)
    if depth_limit_paths:
        return [
            _issue(
                "invalid_generated_test_contract",
                subtask_id=subtask_id,
                message=(
                    "generated_test_contract contains `_depth_limit` placeholder "
                    f"value(s) at {', '.join(depth_limit_paths[:4])}. Replace "
                    "them with concrete JSON example values, or omit "
                    "interface_model and source oracle_claims from "
                    "task_requirement/reference_behavior."
                ),
                contract_path="proof.generated_test_contract",
                invalid_values=depth_limit_paths[:8],
                required_deltas=[
                    {
                        "op": "replace",
                        "path": "proof.generated_test_contract",
                        "replacement": {
                            "interface_model": {},
                            "proof_budget": {
                                "max_generated_tests_per_subtask": (
                                    _DEFAULT_MAX_GENERATED_TESTS_PER_SUBTASK
                                ),
                                "allow_expanded_generated_tests": False,
                            },
                            "oracle_claims": [],
                        },
                    }
                ],
                evidence_refs=evidence_refs,
            )
        ]
    claims = [item for item in contract.get("oracle_claims") or () if isinstance(item, dict)]
    if not claims:
        return [
            _issue(
                "invalid_generated_test_contract",
                subtask_id=subtask_id,
                message="generated_test_contract must include oracle_claims[].",
                required_deltas=[
                    {
                        "op": "add",
                        "path": "proof.generated_test_contract.oracle_claims",
                        "replacement": [],
                    }
                ],
                evidence_refs=evidence_refs,
            )
        ]

    issues: list[ContractIssue] = []
    budget = contract.get("proof_budget") if isinstance(contract.get("proof_budget"), dict) else {}
    max_claims = int(
        budget.get("max_generated_tests_per_subtask") or _DEFAULT_MAX_GENERATED_TESTS_PER_SUBTASK
    )
    allow_expanded = bool(budget.get("allow_expanded_generated_tests"))
    if len(claims) > max_claims and not allow_expanded:
        issues.append(
            _issue(
                "invalid_generated_test_contract",
                subtask_id=subtask_id,
                message=(
                    "Generated oracle exceeds the default proof budget: "
                    f"{len(claims)} claims > {max_claims}. Add a typed "
                    "proof_budget override or split/escalate proof layers after "
                    "core proofs pass."
                ),
                invalid_values=[f"oracle_claim_count:{len(claims)}"],
                required_deltas=[
                    {
                        "op": "replace",
                        "path": "proof.generated_test_contract.proof_budget",
                        "replacement": {
                            "max_generated_tests_per_subtask": max_claims,
                            "allow_expanded_generated_tests": True,
                            "override_reason": "complex_workspace_policy",
                        },
                    }
                ],
                evidence_refs=evidence_refs,
            )
        )
    issues.extend(
        ungrounded_required_property_issues(
            required_properties=tuple(_required_properties(contract)),
            grounding_contract=contract,
            subtask_id=subtask_id,
            evidence_refs=evidence_refs,
        )
    )

    seen_ids: set[str] = set()
    for index, claim in enumerate(claims, start=1):
        claim_id = _claim_id(claim, index)
        if not str(claim.get("claim_id") or "").strip():
            issues.append(
                _issue(
                    "invalid_generated_test_contract",
                    subtask_id=subtask_id,
                    message="Every generated oracle claim needs a stable claim_id.",
                    invalid_values=[f"claim_index:{index}"],
                    required_deltas=[
                        {
                            "op": "replace",
                            "path": "proof.generated_test_contract.oracle_claims",
                            "replacement": claim_id,
                        }
                    ],
                    evidence_refs=evidence_refs,
                )
            )
        elif claim_id in seen_ids:
            issues.append(
                _issue(
                    "invalid_generated_test_contract",
                    subtask_id=subtask_id,
                    message=f"Duplicate generated oracle claim_id `{claim_id}`.",
                    invalid_values=[claim_id],
                    required_deltas=[
                        {
                            "op": "replace",
                            "path": "proof.generated_test_contract.oracle_claims",
                            "values": [claim_id],
                        }
                    ],
                    evidence_refs=evidence_refs,
                )
            )
        seen_ids.add(claim_id)
        source = str(claim.get("source") or "").strip()
        if source not in _VALID_CLAIM_SOURCES:
            issues.append(
                _issue(
                    "invalid_generated_test_contract",
                    subtask_id=subtask_id,
                    message=(f"Generated oracle claim `{claim_id}` lacks a valid source."),
                    invalid_values=[claim_id],
                    required_deltas=[
                        {
                            "op": "replace",
                            "path": "proof.generated_test_contract.oracle_claims",
                            "values": [claim_id],
                        }
                    ],
                    evidence_refs=evidence_refs,
                )
            )

    scoped_claims = _claims_for_failures(claims, failed_node_ids)
    issues.extend(
        _same_input_conflicts(
            scoped_claims,
            subtask_id=subtask_id,
            evidence_refs=evidence_refs,
        )
    )
    issues.extend(
        _interface_domain_conflicts(
            contract,
            scoped_claims,
            subtask_id=subtask_id,
            evidence_refs=evidence_refs,
        )
    )
    return issues


def failed_oracle_contract_issues(
    *,
    generated_contract: Any,
    required_properties: tuple[str, ...] = (),
    failed_node_ids: tuple[str, ...] = (),
    failure_output: str = "",
    subtask_id: str = "",
    evidence_refs: tuple[str, ...] = (),
) -> list[ContractIssue]:
    """Map failed generated proof nodes to typed oracle repair issues."""

    issues: list[ContractIssue] = []
    issues.extend(
        _failed_required_property_issues(
            required_properties=required_properties,
            failed_node_ids=failed_node_ids,
            failure_output=failure_output,
            subtask_id=subtask_id,
            evidence_refs=evidence_refs,
        )
    )
    issues.extend(
        _failed_generated_claim_issues(
            generated_contract,
            failed_node_ids=failed_node_ids,
            subtask_id=subtask_id,
            evidence_refs=evidence_refs,
        )
    )
    if (
        isinstance(generated_contract, dict)
        and generated_contract
        and failed_node_ids
        and not issues
    ):
        issues.append(
            _issue(
                "unmapped_generated_oracle",
                subtask_id=subtask_id,
                message=(
                    "Generated oracle proof failed, but the failed pytest "
                    "node could not be mapped to a stable oracle claim_id or "
                    "required_property. Recovery needs a typed claim/property "
                    "mapping before plan revision."
                ),
                contract_path="proof.generated_test_contract.oracle_claims",
                invalid_values=["unmapped_generated_oracle"],
                required_deltas=[],
                evidence_refs=evidence_refs,
            )
        )
    return issues


def ungrounded_required_property_issues(
    *,
    required_properties: tuple[str, ...],
    grounding_contract: Any = None,
    subtask_id: str = "",
    evidence_refs: tuple[str, ...] = (),
) -> list[ContractIssue]:
    contract = dict(grounding_contract or {}) if isinstance(grounding_contract, dict) else {}
    contract["required_properties"] = list(required_properties)
    return _ungrounded_required_property_issues(
        contract,
        subtask_id=subtask_id,
        evidence_refs=evidence_refs,
    )


def contract_issues_payload(issues: list[ContractIssue]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for issue in issues:
        payload: dict[str, Any] = {
            "code": issue.code,
            "severity": issue.severity,
            "target_subtask_id": issue.target_subtask_id or issue.subtask_id,
            "contract_path": issue.contract_path,
            "invalid_values": list(issue.invalid_values),
            "required_deltas": list(issue.required_deltas),
            "message": issue.message,
        }
        if issue.target_path:
            payload["target_path"] = issue.target_path
        if issue.failure_hash:
            payload["failure_hash"] = issue.failure_hash
        if issue.evidence_refs:
            payload["evidence_refs"] = [
                f"{ref.ref_type}:{ref.ref_id}" for ref in issue.evidence_refs
            ]
        payloads.append({key: value for key, value in payload.items() if value})
    return payloads


def _claims_for_failures(
    claims: list[dict[str, Any]], failed_node_ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    if not failed_node_ids:
        return claims
    failed = {item.replace("\\", "/") for item in failed_node_ids if item}
    matched: list[dict[str, Any]] = []
    for claim in claims:
        refs = {
            str(item).replace("\\", "/")
            for item in (claim.get("test_refs") or [])
            if str(item).strip()
        }
        if refs and refs.intersection(failed):
            matched.append(claim)
    return matched or claims


def _failed_required_property_issues(
    *,
    required_properties: tuple[str, ...],
    failed_node_ids: tuple[str, ...],
    failure_output: str,
    subtask_id: str,
    evidence_refs: tuple[str, ...],
) -> list[ContractIssue]:
    properties = {str(item).strip() for item in required_properties if str(item).strip()}
    if "distinct_inputs_distinct_outputs" not in properties:
        return []
    text = "\n".join([*failed_node_ids, str(failure_output or "")]).lower()
    mentions_property = (
        "distinct_inputs_distinct_outputs" in text
        or "distinct inputs distinct outputs" in text
        or "distinct_outputs" in text
    )
    counterexample_assertion = bool(re.search(r"(?is)\bassert\b.+!=", text))
    if not (mentions_property or counterexample_assertion):
        return []
    return [
        _issue(
            "bad_generated_oracle",
            subtask_id=subtask_id,
            message=(
                "Failed generated proof produced a counterexample to "
                "distinct_inputs_distinct_outputs; this is an ungrounded "
                "oracle property, not an implementation repair."
            ),
            contract_path="proof.oracle.required_properties",
            invalid_values=["distinct_inputs_distinct_outputs"],
            required_deltas=[
                {
                    "op": "remove",
                    "path": "proof.oracle.required_properties",
                    "values": ["distinct_inputs_distinct_outputs"],
                }
            ],
            evidence_refs=evidence_refs,
        )
    ]


def _failed_generated_claim_issues(
    contract: Any,
    *,
    failed_node_ids: tuple[str, ...],
    subtask_id: str,
    evidence_refs: tuple[str, ...],
) -> list[ContractIssue]:
    if not isinstance(contract, dict) or not failed_node_ids:
        return []
    claims = [item for item in contract.get("oracle_claims") or () if isinstance(item, dict)]
    if not claims:
        return []
    failed = {item.replace("\\", "/") for item in failed_node_ids if item}
    claim_ids: list[str] = []
    for index, claim in enumerate(claims, start=1):
        claim_id = _claim_id(claim, index)
        refs = {
            str(item).replace("\\", "/")
            for item in (claim.get("test_refs") or [])
            if str(item).strip()
        }
        claim_needles = {
            claim_id,
            f"test_{claim_id}",
            claim_id.replace("-", "_"),
            f"test_{claim_id.replace('-', '_')}",
        }
        matched_ref = bool(refs and refs.intersection(failed))
        matched_name = any(
            any(needle and needle in node for needle in claim_needles) for node in failed
        )
        if (matched_ref or matched_name) and claim_id not in claim_ids:
            claim_ids.append(claim_id)
    if not claim_ids:
        return []
    return [
        _issue(
            "bad_generated_oracle",
            subtask_id=subtask_id,
            message=(
                "Failed generated proof maps to generated oracle claim(s); "
                "remove or replace the claim(s) in the proof contract."
            ),
            contract_path="proof.generated_test_contract.oracle_claims",
            invalid_values=claim_ids,
            required_deltas=[
                {
                    "op": "remove",
                    "path": "proof.generated_test_contract.oracle_claims",
                    "values": claim_ids,
                }
            ],
            evidence_refs=evidence_refs,
        )
    ]


def _same_input_conflicts(
    claims: list[dict[str, Any]],
    *,
    subtask_id: str,
    evidence_refs: tuple[str, ...],
) -> list[ContractIssue]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for claim in claims:
        subject = _claim_subject(claim)
        if not subject:
            continue
        for input_key in _claim_input_keys(claim):
            grouped.setdefault((subject, input_key), []).append(claim)

    issues: list[ContractIssue] = []
    for (subject, input_key), group in grouped.items():
        acceptance = {
            state for state in (_claim_acceptance(item) for item in group) if state is not None
        }
        outputs = {_json_key(value) for item in group for value in _claim_expected_outputs(item)}
        if acceptance == {True, False}:
            ids = [_claim_id(item, idx) for idx, item in enumerate(group, start=1)]
            issues.append(
                _issue(
                    "bad_generated_oracle",
                    subtask_id=subtask_id,
                    message=(
                        "Generated oracle has contradictory accept/reject "
                        f"claims for `{subject}` with input `{input_key}`."
                    ),
                    invalid_values=[f"{subject}:{input_key}", *ids],
                    required_deltas=[
                        {
                            "op": "remove",
                            "path": "proof.generated_test_contract.oracle_claims",
                            "values": ids,
                        }
                    ],
                    evidence_refs=evidence_refs,
                )
            )
        if len(outputs) > 1 and False not in acceptance:
            ids = [_claim_id(item, idx) for idx, item in enumerate(group, start=1)]
            issues.append(
                _issue(
                    "contradictory_required_behavior",
                    subtask_id=subtask_id,
                    message=(
                        "Generated oracle assigns multiple incompatible "
                        f"outputs to `{subject}` with input `{input_key}`."
                    ),
                    invalid_values=[f"{subject}:{input_key}", *sorted(outputs)],
                    required_deltas=[
                        {
                            "op": "remove",
                            "path": "proof.generated_test_contract.oracle_claims",
                            "values": ids,
                        }
                    ],
                    evidence_refs=evidence_refs,
                )
            )
    return issues


def _interface_domain_conflicts(
    contract: dict[str, Any],
    claims: list[dict[str, Any]],
    *,
    subtask_id: str,
    evidence_refs: tuple[str, ...],
) -> list[ContractIssue]:
    interface = contract.get("interface_model")
    if not isinstance(interface, dict):
        return []
    events = interface.get("events") or interface.get("apis") or []
    domains: dict[str, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        name = str(event.get("name") or event.get("event") or event.get("api") or "").strip()
        if name:
            domains[name] = event

    issues: list[ContractIssue] = []
    for claim in claims:
        subject = _claim_subject(claim)
        domain = domains.get(subject)
        if not domain:
            continue
        accepted = _claim_acceptance(claim)
        if accepted is not True:
            continue
        if _claim_input_case_declared(claim, domain):
            continue
        invalid_hits = [
            value
            for value in _claim_raw_input_values(claim)
            if _value_is_invalid_for_domain(value, domain)
        ]
        if not invalid_hits:
            continue
        claim_id = _claim_id(claim, 1)
        issues.append(
            _issue(
                "oracle_domain_mismatch",
                subtask_id=subtask_id,
                message=(
                    f"Generated oracle claim `{claim_id}` accepts input outside "
                    f"the declared `{subject}` interface domain."
                ),
                invalid_values=[_json_key(item) for item in invalid_hits],
                required_deltas=[
                    {
                        "op": "remove",
                        "path": "proof.generated_test_contract.oracle_claims",
                        "values": [claim_id],
                    }
                ],
                evidence_refs=evidence_refs,
            )
        )
    return issues


def _depth_limit_placeholder_paths(value: Any, path: str = "generated_test_contract") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        if value.get("_depth_limit") is True:
            paths.append(path)
        for key, child in value.items():
            paths.extend(_depth_limit_placeholder_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_depth_limit_placeholder_paths(child, f"{path}[{index}]"))
    return paths


def _ungrounded_required_property_issues(
    contract: dict[str, Any],
    *,
    subtask_id: str,
    evidence_refs: tuple[str, ...],
) -> list[ContractIssue]:
    properties = set(_required_properties(contract))
    if "distinct_inputs_distinct_outputs" not in properties:
        return []
    if _required_property_is_grounded(contract, "distinct_inputs_distinct_outputs"):
        return []
    return [
        _issue(
            "bad_generated_oracle",
            subtask_id=subtask_id,
            message=(
                "Global distinct_inputs_distinct_outputs is not grounded in "
                "the task, interface model, reference behavior, or harness "
                "contract. Many domains legitimately map distinct inputs to "
                "the same output."
            ),
            contract_path="proof.oracle.required_properties",
            invalid_values=["distinct_inputs_distinct_outputs"],
            required_deltas=[
                {
                    "op": "remove",
                    "path": "proof.oracle.required_properties",
                    "values": ["distinct_inputs_distinct_outputs"],
                }
            ],
            evidence_refs=evidence_refs,
        )
    ]


def _required_properties(contract: dict[str, Any]) -> list[str]:
    raw = contract.get("required_properties")
    if raw is None and isinstance(contract.get("oracle"), dict):
        raw = contract["oracle"].get("required_properties")
    return (
        [str(item).strip() for item in (raw or []) if str(item).strip()]
        if isinstance(raw, list)
        else []
    )


def _required_property_is_grounded(contract: dict[str, Any], property_id: str) -> bool:
    for key in (
        "task_requirement",
        "task_requirements",
        "interface_model",
        "reference_behavior",
        "harness_contract",
    ):
        value = contract.get(key)
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True
        if isinstance(value, str) and value.strip():
            return True
    sources = contract.get("required_property_sources")
    if isinstance(sources, dict):
        source = sources.get(property_id)
        if isinstance(source, str) and source.strip():
            return True
        if isinstance(source, list) and any(str(item).strip() for item in source):
            return True
    grounded = contract.get("grounded_required_properties") or contract.get("grounded_properties")
    if isinstance(grounded, list) and property_id in {str(item).strip() for item in grounded}:
        return True
    for claim in contract.get("oracle_claims") or []:
        if not isinstance(claim, dict):
            continue
        if property_id not in {
            str(item).strip()
            for item in (claim.get("grounds_properties") or [])
            if str(item).strip()
        }:
            continue
        if str(claim.get("source") or "").strip() in _VALID_CLAIM_SOURCES:
            return True
    return False


def _claim_id(claim: dict[str, Any], index: int) -> str:
    return str(claim.get("claim_id") or f"claim_{index}").strip()


def _claim_subject(claim: dict[str, Any]) -> str:
    for key in ("subject", "event", "api", "handler", "name"):
        value = str(claim.get(key) or "").strip()
        if value:
            return value
    return ""


def _claim_input_keys(claim: dict[str, Any]) -> list[str]:
    if isinstance(claim.get("input_sequence"), list):
        return ["sequence:" + _json_key(claim.get("input_sequence"))]
    if isinstance(claim.get("input_values"), list):
        return [_json_key(claim.get("input_values"))]
    values = _claim_raw_input_values(claim)
    if not values:
        return ["<unspecified>"]
    return [_json_key(value) for value in values]


def _claim_raw_input_values(claim: dict[str, Any]) -> list[Any]:
    if isinstance(claim.get("input_values"), list):
        return list(claim.get("input_values") or [])
    for key in ("input_value", "input", "value", "argv", "request"):
        if key in claim:
            return [claim.get(key)]
    return []


def _claim_acceptance(claim: dict[str, Any]) -> bool | None:
    for key in ("accepted", "valid"):
        if isinstance(claim.get(key), bool):
            return bool(claim.get(key))
    for key in ("expectation", "expected_behavior", "behavior"):
        value = claim.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
        if normalized in {"accept", "accepted", "valid", "success", "succeeds"}:
            return True
        if normalized in {"reject", "rejected", "invalid", "error", "no-op", "noop"}:
            return False
    return None


def _claim_expected_outputs(claim: dict[str, Any]) -> list[Any]:
    outputs: list[Any] = []
    for key in (
        "expected_output",
        "expected_display",
        "expected_status",
        "expected_result",
        "expected_value",
    ):
        if key in claim:
            outputs.append(claim.get(key))
    return outputs


def _json_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(value)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _value_is_invalid_for_domain(value: Any, domain: dict[str, Any]) -> bool:
    # Explicit out-of-domain declarations are the strong, unambiguous signal:
    # a value the interface lists in `invalid_values` (or that violates a range
    # rule) cannot also be claimed as accepted.
    invalid_values = domain.get("invalid_values")
    if isinstance(invalid_values, list):
        if value in invalid_values or _json_key(value) in {
            _json_key(item) for item in invalid_values
        }:
            return True
        for rule in invalid_values:
            if _range_rule_matches(value, str(rule)):
                return True
    # `valid_values` is treated as an exhaustive allowlist ONLY for scalar enums
    # (e.g. on_digit valid_values=range(10)). For multi-arg interfaces the
    # planner legitimately declares case-objects such as
    # {"inputs": [..], "output": ..} or {"dividend": .., "op": .., "output": ..};
    # those are positive examples, not a denylist. Matching them literally
    # produced false `oracle_domain_mismatch` positives whenever the claim omits
    # the output field or used a different-but-valid input shape, which blocked
    # otherwise-correct plans. Out-of-domain inputs for such interfaces must be
    # declared in `invalid_values` to be flagged.
    valid_values = domain.get("valid_values")
    if (
        isinstance(valid_values, list)
        and valid_values
        and _is_scalar(value)
        and all(_is_scalar(item) for item in valid_values)
    ):
        if value not in valid_values and _json_key(value) not in {
            _json_key(item) for item in valid_values
        }:
            return True
    return False


def _claim_input_case_declared(claim: dict[str, Any], domain: dict[str, Any]) -> bool:
    if not isinstance(claim.get("input_values"), list):
        return False
    claim_values = list(claim.get("input_values") or [])
    valid_values = domain.get("valid_values")
    if not isinstance(valid_values, list) or not valid_values:
        return False
    claim_key = _json_key(claim_values)
    for item in valid_values:
        if _json_key(item) == claim_key:
            return True
        if not isinstance(item, dict):
            continue
        for key in ("inputs", "input_values", "args", "argv"):
            if key in item and _json_key(item.get(key)) == claim_key:
                return True
    return False


def _range_rule_matches(value: Any, rule: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    match = re.fullmatch(r"\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*", rule)
    if not match:
        return False
    op, raw_threshold = match.groups()
    threshold = float(raw_threshold)
    if op == "<":
        return number < threshold
    if op == "<=":
        return number <= threshold
    if op == ">":
        return number > threshold
    return number >= threshold


def _issue(
    code: str,
    *,
    subtask_id: str,
    message: str,
    contract_path: str = "proof.generated_test_contract.oracle_claims",
    invalid_values: list[str] | None = None,
    required_deltas: list[dict[str, Any]] | None = None,
    evidence_refs: tuple[str, ...] = (),
) -> ContractIssue:
    refs: list[EvidenceRef] = []
    for raw in evidence_refs:
        ref_type, _, ref_id = str(raw).partition(":")
        if ref_id:
            refs.append(
                EvidenceRef(
                    ref_type=ref_type,  # type: ignore[arg-type]
                    ref_id=ref_id,
                    produced_by="verifier",
                    subtask_id=subtask_id or None,
                )
            )
    return ContractIssue(
        code=code,
        severity="blocking",
        subtask_id=subtask_id,
        target_subtask_id=subtask_id,
        contract_path=contract_path,
        invalid_values=tuple(invalid_values or ()),
        required_deltas=tuple(dict(item) for item in (required_deltas or ())),
        message=message,
        evidence_refs=tuple(refs),
    )
