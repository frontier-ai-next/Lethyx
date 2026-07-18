"""Contract validators used by Lethyx hard gates."""

import ast

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from lethyx.analysis.shell_commands import validate_argv
from lethyx.contracts.evidence import (
    EvidenceResolver,
    LEDGER_BACKED_EVIDENCE_REF_TYPES,
)
from lethyx.contracts.hashing import diff_hash
from lethyx.contracts.models import (
    CURRENT_CONTRACT_VERSION,
    CompletionContract,
    ContractBundle,
    ContractEnvelope,
    ContractIssue,
    IssueSeverity,
    PlanIR,
    ProofSpec,
    ReviewIssue,
    ReviewContract,
    SubtaskIR,
    TaskRiskProfile,
    VerificationReportRef,
    WorkspaceContext,
)
from lethyx.contracts.runtime_probes import (
    effective_runtime_capabilities,
    proof_requires_capability,
)
from lethyx.contracts.environments import (
    DEFAULT_EXECUTION_ENVIRONMENT_ID,
    find_capability_binding,
)
from lethyx.contracts.contract_paths import (
    InvalidContractPath,
    InvalidContractShape,
    normalize_contract_path,
    validate_contract_delta_shape,
)
from lethyx.contracts.schemas import VALID_REVIEW_CODES
from lethyx.contracts.oracle_validator import (
    BAD_ORACLE_REVIEW_CODES,
    generated_oracle_contract_issues,
    ungrounded_required_property_issues,
)
from lethyx.contracts.layout_policy import (
    is_python_implementation_path,
    validate_plan_layout_policy,
)
from lethyx.contracts.harness_profiles import known_harness_profile_ids
from lethyx.contracts.proof_templates import known_proof_template_ids

_COMPLETION_BLOCKER_PHRASES = (
    "infrastructure blocker",
    "verification blocked",
    "missing source file",
    "missing source files",
    "unresolved blocker",
    "blockers remain",
    "blockers present",
    "blocker still open",
    "blockers blocking",
    "cannot close",
    "cannot complete",
    "cannot mark",
)


ALLOWED_SCHEMA_NAMES = {
    "plan",
    "review",
    "research_summary",
    "completion",
    "verification_report",
}
_ALLOWED_PROOF_KINDS = frozenset(
    {
        "pytest",
        "verification_step",
        "http_boot",
        "behavioral_http",
        "input_sensitivity",
        "mutation_smoke",
        "metamorphic",
        "property_test",
        "import_check",
        "build",
        "command",
    }
)
_CANDIDATE_CONTROL_DIRS = {
    ".git",
    ".memory",
    ".Lethyx",
    ".LETHYX_scratch",
}
_CANDIDATE_CONTROL_ROOT_FILES = {
    "verification.toml",
    "verify.sh",
}
_PRODUCTION_SOURCE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
}
_RUNTIME_READINESS_KEYS = ("readiness", "readiness_probe", "readiness_probes")
_RUNTIME_DRIVER_KEYS = ("assert_command", "interaction_command", "driver_command")
_RUNTIME_FREEFORM_DRIVER_KEYS = (
    "interaction",
    "interaction_test",
    "test_interaction",
    "expected_behavior",
    "expected_evidence",
    "evidence_expectations",
)
_RUNTIME_STARTED_ONLY_PROPERTIES = frozenset(
    {"runtime_started", "module_imports", "build_succeeds", "no_test_tampering"}
)
_PYTEST_PROGRAM_NAMES = frozenset({"pytest", "py.test"})
_HELP_COMMAND_TOKENS = frozenset({"--help", "-h", "/?", "help"})
_LONG_LIVED_PYTHON_SERVER_MODULES = frozenset({"uvicorn", "hypercorn", "flask", "http.server"})
_LONG_LIVED_SERVER_PROGRAMS = frozenset({"uvicorn", "hypercorn", "gunicorn", "flask", "vite"})
_LONG_LIVED_NPM_SCRIPTS = frozenset({"dev", "start", "serve"})
_RUNTIME_HARNESS_PROFILES = frozenset({"desktop_gui_runtime"})
_RUNTIME_CAPABILITY_IDS = frozenset(
    {
        "desktop_gui_runtime",
        "browser_ui",
        "web_ui_browser",
        "http_runtime",
        "http_server",
        "service_runtime",
    }
)
_PYTEST_OPTIONS_WITH_VALUE = frozenset(
    {
        "-k",
        "-m",
        "-o",
        "--basetemp",
        "--confcutdir",
        "--cov",
        "--cov-report",
        "--deselect",
        "--ignore",
        "--ignore-glob",
        "--junitxml",
        "--maxfail",
        "--rootdir",
    }
)
_REVIEW_CODES_REQUIRE_REVISION = {
    "missing_proof",
    "weak_proof",
    "manual_proof",
    "unavailable_proof_target",
    "test_tampering_risk",
    "scope_mismatch",
    "policy_violation",
    "insufficient_research_evidence",
    "requires_human_checkpoint",
    "stale_proof_ref",
    "fake_evidence_ref",
    "invalid_evidence_ref",
    "invalid_python_c_proof",
    "long_lived_server_command",
    "non_ledger_evidence_ref",
    "shell_operator_in_argv",
    "proof_after_patch_missing",
    "proof_scope_mismatch",
    "claim_without_proof",
    "test_tampering_detected",
    "verifier_mutation_attempt",
    "memory_without_verified_evidence",
    "legacy_contract_used",
    "llm_judge_only_evidence",
    "greenfield_python_src_layout_policy",
    "unknown_harness_profile",
    "bad_generated_oracle",
    "plan_contract_issue",
    "inconsistent_generated_oracle",
    "oracle_domain_mismatch",
    "contradictory_required_behavior",
    "invalid_generated_test_contract",
    "unmapped_generated_oracle",
}
_TEMPLATE_ADVISORY_ARGV_CODES = frozenset(
    {
        "import_only_proof",
        "workspace_import_inline_proof",
        "complex_python_inline_proof",
    }
)
_MATERIALIZATION_DIR_SKIP_NAMES = frozenset(
    {
        ".git",
        ".memory",
        ".Lethyx",
        ".LETHYX_scratch",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)


def _workspace_path_materialized(target: Path) -> bool:
    try:
        if target.is_file():
            return target.stat().st_size > 0
        if not target.is_dir():
            return False
        for child in target.rglob("*"):
            try:
                rel_parts = child.relative_to(target).parts
            except ValueError:
                rel_parts = child.parts
            if any(part in _MATERIALIZATION_DIR_SKIP_NAMES for part in rel_parts):
                continue
            if child.is_file() and child.stat().st_size > 0:
                return True
    except OSError:
        return False
    return False


def validate_envelope(envelope: ContractEnvelope) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    if envelope.schema_name not in ALLOWED_SCHEMA_NAMES:
        issues.append(
            ContractIssue(
                code="unknown_contract_schema",
                severity="blocking",
                phase=envelope.phase,
                message=f"Unknown contract schema `{envelope.schema_name}`.",
            )
        )
    if envelope.schema_version != CURRENT_CONTRACT_VERSION:
        issues.append(
            ContractIssue(
                code="unknown_contract_version",
                severity="blocking",
                phase=envelope.phase,
                message=(
                    f"Unsupported contract version `{envelope.schema_version}`; "
                    f"expected `{CURRENT_CONTRACT_VERSION}`."
                ),
            )
        )
    return issues


def _issue(
    code: str,
    message: str,
    *,
    phase: str = "",
    subtask_id: str = "",
    severity: IssueSeverity = "blocking",
) -> ContractIssue:
    return ContractIssue(
        code=code,
        severity=severity,
        phase=phase,
        subtask_id=subtask_id,
        message=message,
    )


def _iter_contract_text(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        if value.strip():
            yield value
        return
    if isinstance(value, (int, float, bool)):
        yield str(value)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.strip():
                yield key
            yield from _iter_contract_text(item)
        return
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _iter_contract_text(item)


def _runtime_readiness_value(options: dict[str, Any]) -> Any:
    for key in _RUNTIME_READINESS_KEYS:
        if key in options:
            return options.get(key)
    return None


def _runtime_readiness_issue(value: Any) -> str:
    if value is None:
        return ""
    specs: list[Any]
    if isinstance(value, dict):
        specs = [value]
    elif isinstance(value, list):
        specs = value
    elif isinstance(value, str):
        return (
            "desktop_gui_runtime readiness must be a structured object/list, "
            "not free-form text. Use {'type':'process_alive'} or "
            "{'type':'log_contains','text':'READY'}; a plain string is "
            "interpreted as log text and can create false timeouts."
        )
    else:
        return "desktop_gui_runtime readiness must be an object or list of objects."
    allowed = {
        "process_alive",
        "alive",
        "wait",
        "wait_seconds",
        "log_contains",
        "stdout_contains",
        "log_regex",
        "stdout_regex",
    }
    for spec in specs:
        if not isinstance(spec, dict):
            return "desktop_gui_runtime readiness entries must be objects."
        kind = str(spec.get("type") or spec.get("kind") or "process_alive").strip()
        if kind not in allowed:
            return (
                f"desktop_gui_runtime readiness type `{kind}` is not supported; "
                "use process_alive, wait_seconds, log_contains, or log_regex."
            )
        if (
            kind in {"log_contains", "stdout_contains"}
            and not str(spec.get("text") or spec.get("contains") or "").strip()
        ):
            return "desktop_gui_runtime log_contains readiness requires `text`."
        if (
            kind in {"log_regex", "stdout_regex"}
            and not str(spec.get("pattern") or spec.get("regex") or "").strip()
        ):
            return "desktop_gui_runtime log_regex readiness requires `pattern`."
    return ""


def _runtime_driver_command(options: dict[str, Any]) -> Any:
    for key in _RUNTIME_DRIVER_KEYS:
        value = options.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, (list, tuple)) and any(str(item).strip() for item in value):
            return value
    return None


def _runtime_driver_is_placeholder(command: Any) -> bool:
    if isinstance(command, str):
        text = command
    elif isinstance(command, (list, tuple)):
        text = " ".join(str(item) for item in command)
    else:
        return False
    normalized = " ".join(text.lower().split())
    return "runtime smoke ok" in normalized or "assert true" in normalized


def _argv_program_name(token: Any) -> str:
    text = str(token or "").strip().replace("\\", "/")
    name = text.rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _command_is_bare_long_lived_server(command: Iterable[Any]) -> bool:
    tokens = [str(item or "").strip() for item in command]
    programs = [_argv_program_name(item) for item in tokens]
    if not programs:
        return False
    if programs[0] in _LONG_LIVED_SERVER_PROGRAMS:
        return True
    if len(programs) >= 3 and programs[0] in {"python", "python3", "py"}:
        if programs[1] == "-m" and programs[2] in _LONG_LIVED_PYTHON_SERVER_MODULES:
            return True
        if programs[1].endswith("manage.py") and programs[2] == "runserver":
            return True
    if len(programs) >= 3 and programs[0] in {"npm", "pnpm", "yarn"}:
        if programs[1] == "run" and programs[2] in _LONG_LIVED_NPM_SCRIPTS:
            return True
        if programs[1] in _LONG_LIVED_NPM_SCRIPTS:
            return True
    if len(programs) >= 2 and programs[0] == "next" and programs[1] == "dev":
        return True
    return False


def _command_invokes_pytest(command: Iterable[Any]) -> bool:
    tokens = [_argv_program_name(token) for token in command]
    for index, token in enumerate(tokens):
        if token in _PYTEST_PROGRAM_NAMES:
            return True
        if token == "-m" and index + 1 < len(tokens) and tokens[index + 1] in _PYTEST_PROGRAM_NAMES:
            return True
    return False


def _command_is_help_invocation(command: Iterable[Any]) -> bool:
    return any(str(token or "").strip().lower() in _HELP_COMMAND_TOKENS for token in command)


def _proof_has_runtime_harness_or_capability(proof: ProofSpec) -> bool:
    if proof.harness_profile in _RUNTIME_HARNESS_PROFILES:
        return True
    if proof.execution.kind in {"http_boot", "behavioral_http"}:
        return True
    caps = {str(item).strip().lower() for item in proof.required_capabilities}
    return bool(caps & _RUNTIME_CAPABILITY_IDS)


def validate_proof_spec(
    proof: ProofSpec,
    *,
    phase: str = "",
    subtask_id: str = "",
    resolver: EvidenceResolver | None = None,
    runtime_capabilities: dict[str, bool] | None = None,
    drive_root: Path | None = None,
) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    execution = proof.execution
    oracle = proof.oracle
    scope = proof.scope
    anti = proof.anti_gaming
    caps = runtime_capabilities or {}
    execution_command_text = "\n".join(str(item) for item in execution.command).lower()
    if execution.kind not in _ALLOWED_PROOF_KINDS:
        issues.append(
            _issue(
                "unknown_proof_kind",
                (
                    f"Unknown proof.execution.kind `{execution.kind}`; "
                    "subtask proofs must use a typed ProofSpec proof kind. "
                    "Workspace verification-only checks such as file_exists "
                    "cannot replace behavioral proof."
                ),
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    if proof.harness_profile and proof.harness_profile not in known_harness_profile_ids():
        issues.append(
            _issue(
                "unknown_harness_profile",
                (
                    f"Unknown proof.harness_profile `{proof.harness_profile}`; "
                    "use a known harness profile id or omit the field."
                ),
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    if proof.template_id and proof.template_id not in known_proof_template_ids():
        issues.append(
            _issue(
                "unknown_proof_template",
                (
                    f"Unknown proof.template_id `{proof.template_id}`; use a "
                    "known canonical proof template id or omit the field."
                ),
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    if proof.harness_profile == "desktop_gui_runtime":
        env_id = str(
            proof.execution.execution_environment_id
            or proof.harness_options.get("execution_environment_id")
            or DEFAULT_EXECUTION_ENVIRONMENT_ID
        ).strip()
        options = proof.harness_options if isinstance(proof.harness_options, dict) else {}
        if execution.kind != "command":
            issues.append(
                _issue(
                    "weak_proof",
                    (
                        "desktop_gui_runtime proof.execution.kind must be "
                        "`command` because run_subtask_proof manages the GUI "
                        "launch lifecycle directly. Use pytest for headless "
                        "controller proof, not for the managed runtime launch."
                    ),
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        if _command_invokes_pytest(execution.command):
            issues.append(
                _issue(
                    "weak_proof",
                    (
                        "desktop_gui_runtime proof.execution.command must be "
                        "the direct managed launch command, not pytest. Put "
                        "post-readiness pytest or driver checks in "
                        "harness_options.assert_command, interaction_command, "
                        "or driver_command."
                    ),
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        if options.get("managed_runtime") is not True:
            issues.append(
                _issue(
                    "weak_proof",
                    (
                        "desktop_gui_runtime proof.harness_options must set "
                        "managed_runtime=true so launch, readiness, driver "
                        "evidence, and cleanup are owned by run_subtask_proof."
                    ),
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        required = {str(item).strip().lower() for item in proof.required_capabilities}
        missing_required = [
            tag for tag in ("desktop_gui_runtime", "subprocess") if tag not in required
        ]
        if missing_required:
            issues.append(
                _issue(
                    "capability_probe_failed",
                    (
                        "desktop_gui_runtime proof requires explicit "
                        "proof.required_capabilities for "
                        + ", ".join(missing_required)
                        + ". Research must declare/probe display automation "
                        "availability before planning real native GUI proof."
                    ),
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        if drive_root is not None:
            binding = find_capability_binding(
                drive_root,
                capability_id="desktop_gui_runtime",
                env_id=env_id,
            )
            if binding is None or not binding.available:
                issues.append(
                    _issue(
                        "capability_probe_failed",
                        (
                            "desktop_gui_runtime proof requires an available "
                            "desktop_gui_runtime capability binding for the "
                            f"same execution environment `{env_id}`. Import-only "
                            "or another interpreter's probe is not sufficient."
                        ),
                        phase=phase,
                        subtask_id=subtask_id,
                    )
                )
        if not proof.harness_options:
            issues.append(
                _issue(
                    "weak_proof",
                    (
                        "desktop_gui_runtime proof must include "
                        "proof.harness_options describing launch, interaction "
                        "driver, evidence, timeout, and cleanup expectations."
                    ),
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        if (
            "subprocess." in execution_command_text
            or "from subprocess import" in execution_command_text
        ):
            issues.append(
                _issue(
                    "weak_proof",
                    (
                        "desktop_gui_runtime proof.execution.command must be "
                        "the direct managed launch command. Do not wrap the "
                        "GUI launch in an inline subprocess; put post-readiness "
                        "checks in harness_options.assert_command, "
                        "interaction_command, or driver_command."
                    ),
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        readiness_value = _runtime_readiness_value(options)
        if readiness_value is None:
            issues.append(
                _issue(
                    "weak_proof",
                    (
                        "desktop_gui_runtime proof.harness_options must include "
                        "structured readiness such as {'type':'process_alive'} "
                        "or {'type':'log_contains','text':'READY'}."
                    ),
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        readiness_issue = _runtime_readiness_issue(readiness_value)
        if readiness_issue:
            issues.append(
                _issue(
                    "weak_proof",
                    readiness_issue,
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        if not any(options.get(key) for key in ("cleanup", "cleanup_command", "teardown")):
            issues.append(
                _issue(
                    "weak_proof",
                    (
                        "desktop_gui_runtime proof.harness_options must include "
                        "cleanup instructions so managed runtime processes/windows "
                        "cannot leak across subtasks."
                    ),
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        required_props = {str(item) for item in oracle.required_properties}
        needs_driver = bool(required_props - _RUNTIME_STARTED_ONLY_PROPERTIES)
        driver_command = _runtime_driver_command(options)
        if needs_driver and not driver_command:
            issues.append(
                _issue(
                    "weak_proof",
                    (
                        "desktop_gui_runtime proof claims behavior beyond "
                        "runtime_started/module_imports/build_succeeds but "
                        "does not provide harness_options.assert_command, "
                        "interaction_command, or driver_command. Put the "
                        "machine interaction/assertion in one of those argv "
                        "fields so run_subtask_proof can execute it after "
                        "readiness."
                    ),
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        elif (
            any(str(key) in options for key in _RUNTIME_FREEFORM_DRIVER_KEYS) and not driver_command
        ):
            issues.append(
                _issue(
                    "weak_proof",
                    (
                        "desktop_gui_runtime proof describes interaction/evidence "
                        "as free-form harness_options text but provides no "
                        "machine driver command. Use assert_command, "
                        "interaction_command, or driver_command as an argv "
                        "command, and keep prose only as supplemental notes."
                    ),
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        elif driver_command and _runtime_driver_is_placeholder(driver_command):
            issues.append(
                _issue(
                    "weak_proof",
                    (
                        "desktop_gui_runtime proof uses a placeholder driver "
                        "command. A runtime smoke can omit behavior claims, but "
                        "any driver/assert/interaction command must exercise "
                        "real user-facing behavior or be removed."
                    ),
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
    if proof.harness_profile == "desktop_gui_headless" and anti.requires_real_runtime:
        issues.append(
            _issue(
                "weak_proof",
                (
                    "desktop_gui_headless proof cannot claim "
                    "anti_gaming.requires_real_runtime=true. Use headless "
                    "adapter/controller proof with requires_real_runtime=false, "
                    "or split out a separate desktop_gui_runtime leaf for real "
                    "window launch evidence."
                ),
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    if (
        anti.requires_real_runtime
        and execution.kind == "command"
        and _command_is_help_invocation(execution.command)
        and not _proof_has_runtime_harness_or_capability(proof)
    ):
        issues.append(
            _issue(
                "contradictory_required_behavior",
                (
                    "proof.anti_gaming.requires_real_runtime=true contradicts "
                    "a help/usage command without a runtime harness profile "
                    "or runtime capability. Use a real launch proof or remove "
                    "the real-runtime claim from this subtask."
                ),
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    if not caps and drive_root is not None:
        caps = effective_runtime_capabilities(drive_root)
    if caps:
        capability_issue = proof_requires_capability(proof, caps)
        if capability_issue:
            issues.append(
                _issue(
                    "capability_probe_failed",
                    capability_issue,
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
    for shell_issue in validate_argv(execution.command, shell=execution.shell):
        if proof.template_id and shell_issue.code in _TEMPLATE_ADVISORY_ARGV_CODES:
            continue
        issues.append(
            _issue(
                shell_issue.code,
                shell_issue.message,
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    if any("--collect-only" in part.lower() for part in execution.command):
        issues.append(
            _issue(
                "collect_only_proof",
                "pytest --collect-only only collects pytest tests; it is not behavioral proof.",
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    if (
        execution.kind in {"behavioral_http", "http_boot"}
        or oracle.oracle_type == "behavioral_http"
    ) and _command_is_bare_long_lived_server(execution.command):
        issues.append(
            _issue(
                "long_lived_server_command",
                (
                    "HTTP behavioral proof command starts a long-lived server directly; "
                    "run_subtask_proof treats proof commands as bounded checks, so this "
                    "will time out after startup. Use a checked-in bounded test/driver "
                    "command (for example pytest/TestClient, Playwright, or an npm test "
                    "script) that starts the server, probes HTTP behavior, and exits."
                ),
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    if anti.requires_real_runtime and anti.allows_mock:
        issues.append(
            _issue(
                "real_runtime_proof_cannot_allow_mock",
                "Real runtime proof cannot allow mocks.",
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    if execution.kind == "pytest" and not (scope.pytest_targets or scope.test_targets):
        issues.append(
            _issue(
                "missing_pytest_targets",
                "pytest proof requires explicit pytest_targets or generic test_targets.",
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    if "no_test_tampering" in oracle.required_properties:
        command_text = " ".join(str(item) for item in execution.command).lower()
        if any(flag in command_text for flag in ("--passwithnotests", "--allowempty")):
            issues.append(
                _issue(
                    "proof_selection_filter_forbidden",
                    "no_test_tampering proofs cannot allow empty test suites.",
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
    if execution.kind == "pytest" and "no_test_tampering" in oracle.required_properties:
        forbidden = frozenset({"-k", "--keyword", "-m", "--deselect", "--ignore", "--ignore-glob"})
        tail_start = 0
        for index, token in enumerate(execution.command):
            if str(token).strip().lower() == "pytest":
                tail_start = index + 1
                break
        for token in execution.command[tail_start:]:
            lowered = str(token).strip().lower()
            if lowered in forbidden or any(
                lowered == prefix or lowered.startswith(prefix + "=")
                for prefix in ("--deselect", "--ignore", "--ignore-glob")
            ):
                issues.append(
                    _issue(
                        "proof_selection_filter_forbidden",
                        "no_test_tampering pytest proofs cannot use -k, -m, "
                        "--deselect, --ignore, or --ignore-glob; fix code or tests instead.",
                        phase=phase,
                        subtask_id=subtask_id,
                    )
                )
                break
    if execution.kind in {"input_sensitivity", "behavioral_http"}:
        has_signal = (
            oracle.input_sensitivity_required
            or "distinct_inputs_distinct_outputs" in oracle.required_properties
            or "invalid_input_rejected" in oracle.required_properties
        )
        if not has_signal:
            issues.append(
                _issue(
                    "missing_behavioral_oracle",
                    "Behavioral proof requires input-sensitivity or negative-case properties.",
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
    issues.extend(
        ungrounded_required_property_issues(
            required_properties=tuple(str(item) for item in oracle.required_properties),
            grounding_contract=proof.generated_test_contract,
            subtask_id=subtask_id,
        )
    )
    if execution.kind == "mutation_smoke" and (
        oracle.oracle_type != "mutation_kill"
        and "mutation_killed" not in oracle.required_properties
    ):
        issues.append(
            _issue(
                "missing_mutation_oracle",
                "mutation_smoke proof requires a mutation-kill oracle.",
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    if proof.human_claims and not oracle.required_properties:
        issues.append(
            _issue(
                "human_claims_without_machine_oracle",
                "Human-readable claims must be backed by machine-checkable required_properties.",
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    if scope.files_under_test and scope.changed_files_expected:
        under_test = {item.replace("\\", "/") for item in scope.files_under_test}
        expected = {item.replace("\\", "/") for item in scope.changed_files_expected}
        if not (under_test & expected):
            issues.append(
                _issue(
                    "proof_scope_mismatch",
                    "files_under_test must overlap changed_files_expected.",
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        non_test_expected = {path for path in expected if not _path_looks_like_test(path)}
        if (
            "no_test_tampering" in oracle.required_properties
            and non_test_expected
            and not (under_test & non_test_expected)
        ):
            issues.append(
                _issue(
                    "proof_scope_mismatch",
                    (
                        "no_test_tampering proof must include at least one "
                        "non-test changed file in files_under_test; test-file "
                        "overlap alone does not prove the runtime artifact."
                    ),
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
    if resolver is not None and proof.evidence_refs:
        issues.extend(
            resolver.validate_refs(proof.evidence_refs, phase=phase, subtask_id=subtask_id)
        )
    return issues


_REVIEW_PHASES_REQUIRE_COVERAGE = frozenset({"plan_review", "subtask_review", "research_review"})


def _review_coverage_message(phase: str) -> str:
    if phase == "research_review":
        return (
            "research_review must include every coverage key with value true. "
            "True means the dimension was evaluated and any unsafe dimension is "
            "represented by a typed blocking issue, including when the dimension "
            "is not directly applicable."
        )
    return (
        "Review must include every coverage key with value true. Typed issues, "
        "not false coverage fields or notes, drive revise/abort decisions."
    )


def _blocking_required_plan_change(change: Any) -> bool:
    if not isinstance(change, dict):
        return False
    severity = str(change.get("severity") or "blocking").strip().lower()
    return severity in {"blocking", "error", "human_required"}


def validate_review_contract(review: ReviewContract, *, phase: str = "") -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    if review.verdict not in {"ok", "revise", "abort"}:
        issues.append(
            _issue(
                "invalid_review_verdict",
                "Review verdict must be ok/revise/abort.",
                phase=phase,
            )
        )
    loop_target = str(review.loop_back_target or "").strip().split(":", 1)[0]
    if phase == "subtask_review" and loop_target == "plan":
        issues.append(
            _issue(
                "invalid_review_loop_back_target",
                (
                    "subtask_review may request implementation revision in execute; "
                    "plan revisions require a typed control-plane "
                    "RecoveryDecision/PlanRevisionPatch."
                ),
                phase=phase,
            )
        )
    if phase in _REVIEW_PHASES_REQUIRE_COVERAGE:
        if review.coverage is None or not review.coverage.is_complete():
            issues.append(
                _issue(
                    "review_coverage_incomplete",
                    _review_coverage_message(phase),
                    phase=phase,
                )
            )
    for item in review.issues:
        if item.code not in VALID_REVIEW_CODES:
            issues.append(
                ContractIssue(
                    code="unknown_review_issue_code",
                    severity="blocking",
                    phase=phase or item.phase,
                    subtask_id=item.subtask_id,
                    message=f"Unknown review issue code `{item.code}`.",
                    evidence_refs=item.evidence_refs,
                )
            )
        if _review_issue_requires_oracle_delta(item):
            contract_path = _review_issue_contract_path(item)
            if item.severity != "blocking":
                issues.append(
                    ContractIssue(
                        code="invalid_bad_oracle_issue_severity",
                        severity="blocking",
                        phase=phase or item.phase,
                        subtask_id=item.subtask_id or item.target_subtask_id,
                        message=(
                            f"{item.code} issues are control-plane blockers "
                            "and must use severity='blocking'."
                        ),
                        evidence_refs=item.evidence_refs,
                    )
                )
            if not contract_path:
                issues.append(
                    ContractIssue(
                        code="bad_oracle_issue_missing_contract_path",
                        severity="blocking",
                        phase=phase or item.phase,
                        subtask_id=item.subtask_id or item.target_subtask_id,
                        message=(
                            f"{item.code} must include contract_path or "
                            "target_path; free-text notes cannot route a plan "
                            "revision."
                        ),
                        evidence_refs=item.evidence_refs,
                    )
                )
            if not item.required_deltas:
                issues.append(
                    ContractIssue(
                        code="bad_oracle_issue_missing_required_deltas",
                        severity="blocking",
                        phase=phase or item.phase,
                        subtask_id=item.subtask_id or item.target_subtask_id,
                        message=(
                            f"{item.code} must include typed required_deltas "
                            "so plan revision can be validated structurally."
                        ),
                        evidence_refs=item.evidence_refs,
                    )
                )
            for delta in item.required_deltas:
                op = str(delta.get("op") or "").strip()
                path = str(delta.get("path") or "").strip()
                try:
                    normalized_delta = validate_contract_delta_shape(delta)
                    path = str(normalized_delta.get("path") or path)
                except (InvalidContractPath, InvalidContractShape) as exc:
                    issues.append(
                        ContractIssue(
                            code="invalid_recovery_contract",
                            severity="blocking",
                            phase=phase or item.phase,
                            subtask_id=item.subtask_id or item.target_subtask_id,
                            message=(
                                "bad-oracle required_delta path is not a "
                                f"canonical PlanIR/ProofContract operation: {exc}"
                            ),
                            evidence_refs=item.evidence_refs,
                        )
                    )
                    continue
                if op not in {"remove", "replace", "add"} or not path:
                    issues.append(
                        ContractIssue(
                            code="invalid_bad_oracle_required_delta",
                            severity="blocking",
                            phase=phase or item.phase,
                            subtask_id=item.subtask_id or item.target_subtask_id,
                            message=(
                                "bad-oracle required_deltas entries need "
                                "op in remove/replace/add and a path."
                            ),
                            evidence_refs=item.evidence_refs,
                        )
                    )
        if item.severity == "human_required" and item.code != "requires_human_checkpoint":
            issues.append(
                ContractIssue(
                    code="invalid_human_required_issue",
                    severity="blocking",
                    phase=phase or item.phase,
                    subtask_id=item.subtask_id,
                    message="human_required severity is reserved for explicit human checkpoints.",
                    evidence_refs=item.evidence_refs,
                )
            )
    ok_blockers = [
        item
        for item in review.issues
        if review.verdict == "ok"
        and (
            item.severity in {"error", "blocking", "human_required"}
            or item.code in _REVIEW_CODES_REQUIRE_REVISION
        )
    ]
    if ok_blockers:
        blocker = ok_blockers[0]
        issues.append(
            _issue(
                "review_ok_with_blocking_issue",
                (
                    "Review verdict `ok` cannot carry typed blocker "
                    f"`{blocker.code}`; use `revise`/`abort` or keep "
                    "nonblocking recommendations in notes."
                ),
                phase=phase or blocker.phase,
                subtask_id=blocker.subtask_id,
            )
        )
    if (
        review.verdict in {"revise", "abort"}
        and not any(
            issue.severity in {"blocking", "human_required", "error"} for issue in review.issues
        )
        and not any(
            _blocking_required_plan_change(change) for change in review.required_plan_changes
        )
    ):
        issues.append(
            _issue(
                "review_without_machine_issue",
                (
                    "revise/abort requires at least one typed error/blocking/"
                    "human_required issue or blocking required_plan_change."
                ),
                phase=phase,
            )
        )
    return issues


def _normalise_candidate_plan_path(value: str) -> str:
    text = str(value or "").strip().strip("`'\"")
    if not text:
        return ""
    text = text.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _is_windows_absolute_path(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[0].isalpha() and value[2] in {"/", "\\"}


def _candidate_plan_path_issue(path: str, *, phase: str, subtask_id: str) -> ContractIssue | None:
    normalised = _normalise_candidate_plan_path(path)
    if not normalised:
        return None
    if normalised.startswith("/") or _is_windows_absolute_path(normalised):
        return _issue(
            "candidate_path_outside_workspace",
            f"Candidate plan path `{path}` must be workspace-relative.",
            phase=phase,
            subtask_id=subtask_id,
        )
    parts = [part for part in normalised.split("/") if part]
    if any(part == ".." for part in parts):
        return _issue(
            "candidate_path_outside_workspace",
            f"Candidate plan path `{path}` must not escape the workspace.",
            phase=phase,
            subtask_id=subtask_id,
        )
    first = parts[0] if parts else ""
    if first == "workspaces":
        hint = "/".join(parts[2:]) if len(parts) >= 3 else ""
        hint_text = f" Use `{hint}` instead." if hint else ""
        return _issue(
            "candidate_path_outside_workspace",
            (
                f"Candidate plan path `{path}` is repository-relative. "
                "Phase plan file paths are already relative to the active "
                "workspace, so do not prefix them with `workspaces/<id>/`."
                f"{hint_text}"
            ),
            phase=phase,
            subtask_id=subtask_id,
        )
    if first in _CANDIDATE_CONTROL_DIRS:
        return _issue(
            "candidate_control_path_forbidden",
            f"Candidate plan path `{path}` targets supervisor/control state.",
            phase=phase,
            subtask_id=subtask_id,
        )
    if len(parts) == 1 and parts[0] in _CANDIDATE_CONTROL_ROOT_FILES:
        return _issue(
            "candidate_control_path_forbidden",
            f"Candidate plan path `{path}` is supervisor/evaluator configuration, not generated project work.",
            phase=phase,
            subtask_id=subtask_id,
        )
    return None


def validate_plan_candidate_paths(plan: PlanIR, *, phase: str = "plan") -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    for subtask in plan.subtasks:
        declared_paths = [
            *subtask.files_to_change,
            *subtask.files_to_create,
        ]
        if subtask.proof is not None:
            declared_paths.extend(subtask.proof.scope.files_under_test)
            declared_paths.extend(subtask.proof.scope.changed_files_expected)
        for path in declared_paths:
            issue = _candidate_plan_path_issue(
                path,
                phase=phase,
                subtask_id=subtask.id,
            )
            if issue is not None:
                issues.append(issue)
    return issues


def _path_looks_like_test(path: str) -> bool:
    normalised = str(path or "").replace("\\", "/").lower().strip("/")
    if not normalised:
        return False
    parts = [part for part in normalised.split("/") if part]
    if any(part in {"test", "tests", "__tests__"} for part in parts[:-1]):
        return True
    name = parts[-1]
    stem = name.rsplit(".", 1)[0]
    return (
        stem.startswith("test_")
        or stem.endswith("_test")
        or name.endswith((".test.js", ".test.jsx", ".test.ts", ".test.tsx"))
        or name.endswith((".spec.js", ".spec.jsx", ".spec.ts", ".spec.tsx"))
    )


def _review_issue_contract_path(item: ReviewIssue) -> str:
    contract_path = str(item.contract_path or "").strip()
    if contract_path:
        try:
            return normalize_contract_path(contract_path).path
        except InvalidContractPath:
            return contract_path
    target_path = str(item.target_path or "").strip()
    if not target_path:
        return ""
    try:
        return normalize_contract_path(target_path).path
    except InvalidContractPath:
        return ""


def _review_issue_requires_oracle_delta(item: ReviewIssue) -> bool:
    if item.code not in BAD_ORACLE_REVIEW_CODES:
        return False
    if item.code == "plan_contract_issue":
        return False
    if item.code != "contradictory_required_behavior":
        return True
    path = _review_issue_contract_path(item)
    if not path:
        return False
    return (
        path == "proof.oracle"
        or path.startswith("proof.oracle.")
        or path == "proof.generated_test_contract"
        or path.startswith("proof.generated_test_contract.")
        or path == "generated_test_contract"
        or path.startswith("generated_test_contract.")
    )


def _pytest_target_file(value: Any) -> str:
    text = _normalise_candidate_plan_path(str(value or "").split("::", 1)[0])
    if not text or text.startswith("-"):
        return ""
    text = text.split("[", 1)[0]
    return text if text.lower().endswith(".py") else ""


def _test_target_file(value: Any) -> str:
    text = _normalise_candidate_plan_path(str(value or "").split("::", 1)[0])
    if not text or text.startswith("-"):
        return ""
    text = text.split("[", 1)[0]
    suffixes = (
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
    )
    return text if text.lower().endswith(suffixes) else ""


def _pytest_targets_from_command(command: Iterable[Any]) -> set[str]:
    tokens = list(command)
    tail_start = -1
    for index, token in enumerate(tokens):
        name = _argv_program_name(token)
        if name in _PYTEST_PROGRAM_NAMES:
            tail_start = index + 1
            break
        if (
            name == "-m"
            and index + 1 < len(tokens)
            and _argv_program_name(tokens[index + 1]) in _PYTEST_PROGRAM_NAMES
        ):
            tail_start = index + 2
            break
    if tail_start < 0:
        return set()

    targets: set[str] = set()
    skip_next = False
    for raw_token in tokens[tail_start:]:
        token = str(raw_token or "").strip()
        if not token:
            continue
        if skip_next:
            skip_next = False
            continue
        option_name = token.split("=", 1)[0].lower()
        if token.startswith("-"):
            if "=" not in token and option_name in _PYTEST_OPTIONS_WITH_VALUE:
                skip_next = True
            continue
        if target := _pytest_target_file(token):
            targets.add(target)
    return targets


def _pytest_target_files_from_proof(proof: ProofSpec) -> set[str]:
    targets = {
        target for item in proof.scope.pytest_targets if (target := _pytest_target_file(item))
    }
    targets.update(
        target for item in proof.scope.test_targets if (target := _test_target_file(item))
    )
    targets.update(_pytest_targets_from_command(proof.execution.command))
    return targets


def _declared_subtask_paths(subtask: SubtaskIR) -> set[str]:
    values: list[Any] = [
        *subtask.files_to_create,
        *subtask.files_to_change,
        *getattr(subtask, "files_affected", ()),
    ]
    if subtask.proof is not None:
        values.extend(subtask.proof.scope.changed_files_expected)
        values.extend(subtask.proof.scope.test_targets)
    return {
        normalized
        for item in values
        if (normalized := (_pytest_target_file(item) or _normalise_candidate_plan_path(str(item))))
    }


def _workspace_relative_file_exists(workspace_root: Path, target: str) -> bool:
    root = workspace_root.resolve()
    candidate = (root / target).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return candidate.is_file()


def validate_plan_pytest_target_scope(
    plan: PlanIR,
    *,
    context: WorkspaceContext | None = None,
    phase: str = "plan",
) -> list[ContractIssue]:
    if context is None or not getattr(context, "workspace_root", None):
        return []
    workspace_root = Path(context.workspace_root)
    if not workspace_root.is_dir():
        return []

    issues: list[ContractIssue] = []
    prior_declared_paths: set[str] = set()
    for subtask in plan.subtasks:
        proof = subtask.proof
        if proof is None:
            prior_declared_paths.update(_declared_subtask_paths(subtask))
            continue
        declared_paths = _declared_subtask_paths(subtask)
        for target in sorted(_pytest_target_files_from_proof(proof)):
            if target in declared_paths:
                continue
            if target in prior_declared_paths:
                continue
            if _workspace_relative_file_exists(workspace_root, target):
                continue
            issues.append(
                ContractIssue(
                    code="proof_scope_mismatch",
                    severity="blocking",
                    phase=phase,
                    subtask_id=subtask.id,
                    target_subtask_id=subtask.id,
                    contract_path="proof.scope.pytest_targets",
                    invalid_values=(target,),
                    required_deltas=(
                        {
                            "op": "add",
                            "path": "files_to_create",
                            "values": [target],
                        },
                    ),
                    message=(
                        "pytest proof targets missing file "
                        f"`{target}`, but the subtask does not declare that "
                        "file in files_to_create, files_to_change, or "
                        "proof.scope.changed_files_expected. Declare the new "
                        "test/driver file or target an existing workspace file."
                    ),
                )
            )
        prior_declared_paths.update(declared_paths)
    return issues


def validate_plan_declared_change_scope(
    plan: PlanIR, *, phase: str = "plan"
) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    for subtask in plan.subtasks:
        proof = subtask.proof
        if proof is None:
            continue
        declared = sorted(
            {
                _normalise_candidate_plan_path(path)
                for path in (
                    *subtask.files_to_create,
                    *subtask.files_to_change,
                    *getattr(subtask, "files_affected", ()),
                )
                if _normalise_candidate_plan_path(path)
            }
        )
        if not declared or proof.scope.changed_files_expected:
            if declared and proof.scope.changed_files_expected:
                expected = {
                    _normalise_candidate_plan_path(path)
                    for path in proof.scope.changed_files_expected
                    if _normalise_candidate_plan_path(path)
                }
                missing = tuple(path for path in declared if path not in expected)
                if missing:
                    issues.append(
                        ContractIssue(
                            code="proof_scope_mismatch",
                            severity="blocking",
                            phase=phase,
                            subtask_id=subtask.id,
                            target_subtask_id=subtask.id,
                            contract_path="proof.scope.changed_files_expected",
                            invalid_values=missing,
                            required_deltas=(
                                {
                                    "op": "add",
                                    "path": "proof.scope.changed_files_expected",
                                    "values": list(missing),
                                },
                            ),
                            message=(
                                "Subtask declares workspace files to "
                                "create/change that are absent from "
                                "proof.scope.changed_files_expected: "
                                + ", ".join(f"`{path}`" for path in missing)
                                + ". Keep files_to_create/files_to_change "
                                "aligned with the typed proof scope so execute "
                                "does not pass a narrow proof and then fail "
                                "materialization on undeclared proof-owned "
                                "files."
                            ),
                        )
                    )
            continue
        issues.append(
            ContractIssue(
                code="proof_scope_mismatch",
                severity="blocking",
                phase=phase,
                subtask_id=subtask.id,
                target_subtask_id=subtask.id,
                contract_path="proof.scope.changed_files_expected",
                required_deltas=(
                    {
                        "op": "replace",
                        "path": "proof.scope.changed_files_expected",
                        "values": declared,
                    },
                ),
                message=(
                    "Subtask declares workspace files to create/change but "
                    "proof.scope.changed_files_expected is empty. List the "
                    "owned changed files in the typed proof scope before plan "
                    "acceptance so execute completion can validate fresh proof "
                    "against the active subtask instead of repairing the plan "
                    "mid-execute."
                ),
            )
        )
    return issues


def validate_plan_test_tampering_policy(
    plan: PlanIR, *, phase: str = "plan"
) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    for subtask in plan.subtasks:
        proof = subtask.proof
        changed_paths = [*subtask.files_to_create, *subtask.files_to_change]
        if proof is not None:
            changed_paths.extend(proof.scope.changed_files_expected)
        test_paths = sorted({path for path in changed_paths if _path_looks_like_test(path)})
        if not test_paths or proof is None:
            continue
        if "no_test_tampering" in proof.oracle.required_properties:
            generated_contract = subtask.generated_test_contract or proof.generated_test_contract
            behavioural_test_paths = [
                path
                for path in test_paths
                if not path.replace("\\", "/").endswith("/__init__.py")
                and path.replace("\\", "/") != "tests/__init__.py"
            ]
            if (
                behavioural_test_paths
                and (proof.execution.kind == "pytest" or proof.scope.test_targets)
                and not generated_contract
            ):
                issues.append(
                    ContractIssue(
                        code="invalid_generated_test_contract",
                        severity="warning",
                        phase=phase,
                        subtask_id=subtask.id,
                        target_subtask_id=subtask.id,
                        contract_path="proof.generated_test_contract",
                        invalid_values=tuple(behavioural_test_paths),
                        required_deltas=(
                            {
                                "op": "add",
                                "path": "proof.generated_test_contract.oracle_claims",
                                "values": [
                                    "claim_id",
                                    "source",
                                    "subject",
                                    "typed_input",
                                    "expected_behavior",
                                    "test_refs",
                                ],
                            },
                        ),
                        message=(
                            "Subtask creates or changes pytest tests under "
                            f"{behavioural_test_paths}; generated tests need a "
                            "typed proof.generated_test_contract with sourced "
                            "oracle_claims and test_refs so contradictory "
                            "oracles can route to plan revision."
                        ),
                    )
                )
            continue
        issues.append(
            _issue(
                "test_tampering_risk",
                (
                    f"Subtask changes test path(s) {test_paths}; add "
                    "`no_test_tampering` to this same subtask's "
                    "proof.oracle.required_properties."
                ),
                phase=phase,
                subtask_id=subtask.id,
            )
        )
    return issues


def _path_extension(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def _is_production_source_path(path: str) -> bool:
    normalised = _normalise_candidate_plan_path(path)
    if not normalised:
        return False
    if not is_python_implementation_path(normalised):
        return False
    return _path_extension(normalised) in _PRODUCTION_SOURCE_EXTENSIONS


def _is_package_init_path(path: str) -> bool:
    normalised = _normalise_candidate_plan_path(path)
    return normalised == "src/__init__.py" or normalised.endswith("/__init__.py")


def _src_layout_package_roots(paths: Iterable[str]) -> set[str]:
    roots: set[str] = set()
    for path in paths:
        normalised = _normalise_candidate_plan_path(path).replace("\\", "/")
        parts = [part for part in normalised.split("/") if part]
        if len(parts) >= 3 and parts[0] == "src" and parts[-1].endswith(".py"):
            roots.add(parts[1])
    return roots


def _proof_env_exposes_src(env: dict[str, str]) -> bool:
    value = str(env.get("PYTHONPATH") or env.get("pythonpath") or "")
    if not value:
        return False
    parts = [
        part.strip().replace("\\", "/").strip("./")
        for chunk in value.split(";")
        for part in chunk.split(":")
    ]
    return any(part == "src" or part.endswith("/src") for part in parts)


def _inline_python_code_from_command(command: Iterable[Any]) -> str:
    tokens = [str(item) for item in command]
    for idx, token in enumerate(tokens[:-1]):
        if token.strip().lower() == "-c":
            return tokens[idx + 1]
    return ""


def _parse_inline_python_command(command: Iterable[Any]) -> ast.AST | None:
    code = _inline_python_code_from_command(command)
    if not code:
        return None
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _ast_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _ast_call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _ast_string_literal(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _ast_is_sys_path(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return _ast_call_name(node) == "sys.path"
    if isinstance(node, ast.Subscript):
        return _ast_is_sys_path(node.value)
    return False


def _module_matches_package(module: str, package: str) -> bool:
    module_l = str(module or "").strip().lower()
    package_l = package.lower()
    return module_l == package_l or module_l.startswith(f"{package_l}.")


def _proof_command_exposes_src(command: Iterable[Any]) -> bool:
    tree = _parse_inline_python_command(command)
    if tree is None:
        return False

    def exposes(value: ast.AST) -> bool:
        text = _ast_string_literal(value).replace("\\", "/").strip().strip("./")
        return text == "src" or text.endswith("/src")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = _ast_call_name(node.func)
            if call_name in {"sys.path.insert", "sys.path.append", "site.addsitedir"}:
                if any(exposes(arg) for arg in node.args):
                    return True
        elif isinstance(node, ast.Assign):
            if not any(_ast_is_sys_path(target) for target in node.targets):
                continue
            value = node.value
            candidates = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            if any(exposes(item) for item in candidates):
                return True
        elif isinstance(node, ast.AugAssign):
            if not _ast_is_sys_path(node.target):
                continue
            value = node.value
            candidates = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            if any(exposes(item) for item in candidates):
                return True
    return False


def _proof_command_imports_package(command: Iterable[Any], package: str) -> bool:
    tree = _parse_inline_python_command(command)
    if tree is None:
        return False
    import_call_names = {
        "__import__",
        "import_module",
        "importlib.import_module",
        "find_spec",
        "util.find_spec",
        "importlib.util.find_spec",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_module_matches_package(alias.name, package) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and _module_matches_package(node.module, package):
                return True
        elif isinstance(node, ast.Call):
            call_name = _ast_call_name(node.func)
            if call_name not in import_call_names or not node.args:
                continue
            module = _ast_string_literal(node.args[0])
            if module and _module_matches_package(module, package):
                return True
    return False


def _proof_command_runs_package_module(command: Iterable[Any], package: str) -> bool:
    tokens = [str(item).strip().strip("'\"").lower() for item in command]
    package_l = package.lower()
    for idx, token in enumerate(tokens[:-1]):
        if token != "-m":
            continue
        module = tokens[idx + 1].strip("'\"")
        if module == package_l or module.startswith(f"{package_l}."):
            return True
    return False


def _python_m_module_targets(command: Iterable[Any]) -> tuple[str, ...]:
    targets: list[str] = []
    tokens = [str(item or "").strip().strip("'\"") for item in command]
    for idx, token in enumerate(tokens[:-1]):
        if token != "-m":
            continue
        module = tokens[idx + 1].strip().strip("'\"")
        if not module:
            continue
        if _argv_program_name(module) in _PYTEST_PROGRAM_NAMES:
            continue
        parts = module.split(".")
        if any(not part.isidentifier() for part in parts):
            continue
        targets.append(module)
    return tuple(targets)


def _src_layout_package_bases(paths: Iterable[str]) -> dict[str, set[str]]:
    bases: dict[str, set[str]] = {}
    for path in paths:
        normalised = _normalise_candidate_plan_path(path).replace("\\", "/")
        parts = [part for part in normalised.split("/") if part]
        if len(parts) >= 3 and parts[0] == "src" and parts[-1].endswith(".py"):
            bases.setdefault(parts[1], set()).add("src")
        elif len(parts) >= 4 and parts[1] == "src" and parts[-1].endswith(".py"):
            bases.setdefault(parts[2], set()).add(f"{parts[0]}/src")
    return bases


def _plan_src_layout_package_bases(plan: PlanIR) -> dict[str, set[str]]:
    paths: list[str] = []
    for subtask in plan.subtasks:
        paths.extend(str(path) for path in subtask.files_to_create)
        paths.extend(str(path) for path in subtask.files_to_change)
        paths.extend(str(path) for path in getattr(subtask, "files_affected", ()))
        if subtask.proof is None:
            continue
        paths.extend(str(path) for path in subtask.proof.scope.files_under_test)
        paths.extend(str(path) for path in subtask.proof.scope.changed_files_expected)
    return _src_layout_package_bases(paths)


def _src_layout_module_entrypoint_candidates(
    module: str, *, package: str, base: str
) -> tuple[str, ...]:
    module_parts = [part for part in module.split(".") if part]
    package_l = package.lower()
    if not module_parts or module_parts[0].lower() != package_l:
        return ()
    if len(module_parts) == 1:
        return (f"{base}/{package}/__main__.py",)
    rel = "/".join(module_parts[1:])
    return (
        f"{base}/{package}/{rel}.py",
        f"{base}/{package}/{rel}/__main__.py",
    )


def validate_plan_python_module_entrypoint_scope(
    plan: PlanIR,
    *,
    context: WorkspaceContext | None = None,
    phase: str = "plan",
) -> list[ContractIssue]:
    package_bases = _plan_src_layout_package_bases(plan)
    if not package_bases:
        return []

    workspace_root: Path | None = None
    if context is not None and getattr(context, "workspace_root", None):
        root = Path(context.workspace_root)
        if root.is_dir():
            workspace_root = root

    issues: list[ContractIssue] = []
    prior_declared_paths: set[str] = set()
    for subtask in plan.subtasks:
        proof = subtask.proof
        declared_paths = _declared_subtask_paths(subtask)
        available_paths = set(prior_declared_paths)
        available_paths.update(declared_paths)
        if proof is None:
            prior_declared_paths.update(declared_paths)
            continue

        for module in _python_m_module_targets(proof.execution.command):
            module_l = module.lower()
            for package in sorted(package_bases):
                package_l = package.lower()
                if module_l != package_l and not module_l.startswith(f"{package_l}."):
                    continue
                candidates: list[str] = []
                for base in sorted(package_bases[package]):
                    candidates.extend(
                        _src_layout_module_entrypoint_candidates(module, package=package, base=base)
                    )
                if not candidates:
                    continue
                if any(candidate in available_paths for candidate in candidates):
                    continue
                if workspace_root is not None and any(
                    _workspace_relative_file_exists(workspace_root, candidate)
                    for candidate in candidates
                ):
                    continue

                required = candidates[0]
                issues.append(
                    ContractIssue(
                        code="proof_scope_mismatch",
                        severity="blocking",
                        phase=phase,
                        subtask_id=subtask.id,
                        target_subtask_id=subtask.id,
                        contract_path="proof.execution.command",
                        invalid_values=(module, required),
                        required_deltas=(
                            {
                                "op": "add",
                                "path": "files_to_create",
                                "values": [required],
                            },
                            {
                                "op": "add",
                                "path": "proof.scope.changed_files_expected",
                                "values": [required],
                            },
                        ),
                        message=(
                            "Proof command launches src-layout module "
                            f"`python -m {module}`, but no matching runnable "
                            "entrypoint is declared in this subtask, an earlier "
                            "subtask, or the current workspace. Declare "
                            f"`{required}` (for package launches, this is "
                            "`__main__.py`) or launch a concrete module file "
                            "that is already in scope."
                        ),
                    )
                )
                break
        prior_declared_paths.update(declared_paths)
    return issues


def _plan_src_layout_package_roots(plan: PlanIR) -> set[str]:
    paths: list[str] = []
    for subtask in plan.subtasks:
        paths.extend(str(path) for path in subtask.files_to_create)
        paths.extend(str(path) for path in subtask.files_to_change)
        paths.extend(str(path) for path in getattr(subtask, "files_affected", ()))
        if subtask.proof is None:
            continue
        paths.extend(str(path) for path in subtask.proof.scope.files_under_test)
        paths.extend(str(path) for path in subtask.proof.scope.changed_files_expected)
    return _src_layout_package_roots(paths)


def validate_plan_src_layout_command_proofs(
    plan: PlanIR,
    *,
    phase: str = "plan",
) -> list[ContractIssue]:
    src_packages = _plan_src_layout_package_roots(plan)
    if not src_packages:
        return []

    issues: list[ContractIssue] = []
    for subtask in plan.subtasks:
        proof = subtask.proof
        if proof is None:
            continue
        if proof.template_id:
            continue
        command = list(proof.execution.command)
        if _proof_env_exposes_src(proof.execution.env) or _proof_command_exposes_src(command):
            continue
        for package in sorted(src_packages):
            if not (
                _proof_command_imports_package(command, package)
                or _proof_command_runs_package_module(command, package)
            ):
                continue
            issues.append(
                _issue(
                    "package_import_env_mismatch",
                    (
                        f"Proof command imports src-layout package `{package}` "
                        "without exposing `src` on sys.path. Set "
                        "`proof.execution.env.PYTHONPATH` to `src` (or add an "
                        "equivalent sys.path insertion) before plan acceptance; "
                        "pytest configuration does not affect command proofs."
                    ),
                    phase=phase,
                    subtask_id=subtask.id,
                )
            )
            break
    return issues


def validate_subtask_proof_strength(
    subtask: SubtaskIR,
    *,
    phase: str = "plan",
) -> list[ContractIssue]:
    if subtask.proof is None:
        return []
    if subtask.proof.template_id:
        return []
    declared_paths = (*subtask.files_to_create, *subtask.files_to_change)
    src_packages = _src_layout_package_roots(declared_paths)
    command = list(subtask.proof.execution.command)
    if (
        src_packages
        and not _proof_env_exposes_src(subtask.proof.execution.env)
        and not _proof_command_exposes_src(command)
    ):
        for package in sorted(src_packages):
            if _proof_command_imports_package(
                command, package
            ) or _proof_command_runs_package_module(command, package):
                return [
                    _issue(
                        "package_import_env_mismatch",
                        (
                            f"Proof command imports src-layout package `{package}` "
                            "without exposing `src` on sys.path. Add "
                            "`sys.path.insert(0, 'src')` to the command or set "
                            "`proof.execution.env.PYTHONPATH` to `src` before "
                            "plan acceptance, so execute does not fail on a "
                            "known packaging import mismatch."
                        ),
                        phase=phase,
                        subtask_id=subtask.id,
                    )
                ]
    production_paths = [
        path
        for path in declared_paths
        if _is_production_source_path(path) and not _is_package_init_path(path)
    ]
    if production_paths and subtask.proof.execution.kind == "import_check":
        return [
            _issue(
                "weak_proof",
                (
                    "Production source leaves cannot use import-only proof. "
                    "Use pytest, input-sensitivity, metamorphic, mutation, "
                    "property, HTTP, or another behavioral oracle tied to the "
                    "changed source."
                ),
                phase=phase,
                subtask_id=subtask.id,
            )
        ]
    return []


def validate_plan_capability_consistency(
    plan: PlanIR,
    declaration: Any,
    *,
    phase: str = "plan",
) -> list[ContractIssue]:
    # Capability availability is enforced by the structured probe audit at
    # submit_capability_declaration time (runtime evidence), not by scanning
    # plan prose for words that look like they contradict the declaration.
    return []


def _completion_text_blobs(
    completion: CompletionContract,
    *,
    raw_completion: dict | None,
) -> str:
    parts: list[str] = [str(completion.notes or "")]
    if isinstance(raw_completion, dict):
        for key in ("summary", "notes", "status_notes"):
            value = raw_completion.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)
    for claim in completion.completed_claims:
        parts.append(str(claim.text or ""))
    return "\n".join(parts)


def _completion_has_unresolved_blocker_language(text: str) -> bool:
    blob = str(text or "").strip().lower()
    if not blob:
        return False
    if any(phrase in blob for phrase in _COMPLETION_BLOCKER_PHRASES):
        return True
    for verb in ("cannot close", "cannot complete", "cannot mark"):
        if verb in blob and ("done" in blob or "complete" in blob):
            return True
    return False


def _normalised_string_list(value: Any) -> set[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
    else:
        values = []
    return {norm for item in values if (norm := _normalise_candidate_plan_path(str(item)))}


def validate_completion_materialization(
    completion: CompletionContract,
    *,
    active_subtask: dict | None,
    workspace_root: str,
    raw_completion: dict | None = None,
    phase: str = "",
) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    if active_subtask is None:
        return issues
    subtask_id = str(active_subtask.get("id") or completion.subtask_id or "")
    declared_created = _normalised_string_list(active_subtask.get("files_to_create"))
    declared_deleted = _normalised_string_list(active_subtask.get("files_to_delete"))
    declared: set[str] = set()
    for key in ("files_to_create", "files_to_change", "files_affected"):
        declared.update(_normalised_string_list(active_subtask.get(key)))
    proof = active_subtask.get("proof")
    if isinstance(proof, dict):
        scope = proof.get("scope")
        if isinstance(scope, dict):
            for key in ("files_under_test", "changed_files_expected"):
                items = scope.get(key)
                if isinstance(items, (list, tuple)):
                    for item in items:
                        norm = _normalise_candidate_plan_path(str(item))
                        if norm:
                            declared.add(norm)
    declared.update(declared_deleted)

    blockers = []
    if isinstance(raw_completion, dict):
        raw_blockers = raw_completion.get("blockers")
        if isinstance(raw_blockers, list):
            blockers = [str(item) for item in raw_blockers if str(item).strip()]
    status_done = str(completion.status or "").lower() in {"done", "ok", "complete"}
    if blockers and status_done:
        issues.append(
            _issue(
                "completion_blocked_not_done",
                "Completion cannot be marked done while blockers are present.",
                phase=phase,
                subtask_id=subtask_id,
            )
        )
    elif status_done and _completion_has_unresolved_blocker_language(
        _completion_text_blobs(completion, raw_completion=raw_completion)
    ):
        issues.append(
            _issue(
                "completion_blocked_not_done",
                "Completion cannot be marked done while notes or claims still describe "
                "unresolved blockers (for example infrastructure blocker, verification "
                "blocked, or missing source files). Use typed `blockers` or keep the "
                "subtask pending until the blocker is resolved.",
                phase=phase,
                subtask_id=subtask_id,
            )
        )

    deleted = set(completion.deleted_files)
    if isinstance(raw_completion, dict):
        for key in ("deleted_files", "removed_files", "expected_absent_files"):
            deleted.update(_normalised_string_list(raw_completion.get(key)))
    deleted.update(declared_deleted)
    deleted = {norm for path in deleted if (norm := _normalise_candidate_plan_path(str(path)))}
    root = Path(workspace_root) if workspace_root else None
    changed = [
        _normalise_candidate_plan_path(path)
        for path in completion.changed_files
        if _normalise_candidate_plan_path(path)
    ]
    if declared and changed:
        if not any(path in declared for path in changed):
            issues.append(
                _issue(
                    "scope_mismatch",
                    "Completion changed_files must overlap the active subtask declared scope.",
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
        extra = sorted({path for path in changed if path not in declared})
        if extra:
            issues.append(
                _issue(
                    "scope_mismatch",
                    "Completion changed_files include paths outside the active subtask "
                    f"declared scope: {', '.join(extra)}.",
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )

    for path in active_subtask.get("files_to_create") or []:
        norm = _normalise_candidate_plan_path(str(path))
        if not norm or not root:
            continue
        target = root / norm
        if not _workspace_path_materialized(target):
            issues.append(
                _issue(
                    "subtask_materialization_missing",
                    f"Declared created path `{norm}` is missing or empty on disk.",
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )

    for path in deleted:
        if not root:
            continue
        target = root / path
        if target.exists():
            issues.append(
                _issue(
                    "subtask_materialization_present",
                    f"Completion declares file `{path}` deleted, but it still exists on disk.",
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )

    for path in changed:
        if not root or path in deleted or path not in declared_created:
            continue
        target = root / path
        if not _workspace_path_materialized(target):
            issues.append(
                _issue(
                    "subtask_materialization_missing",
                    f"Completion changed path `{path}` is missing or empty on disk.",
                    phase=phase,
                    subtask_id=subtask_id,
                )
            )
    return issues


def validate_done_subtasks_materialized(
    *,
    subtasks: list[Any],
    workspace_root: str,
    phase: str = "execute",
) -> list[ContractIssue]:
    """Block advancing execute when a done subtask still lacks declared files on disk."""
    issues: list[ContractIssue] = []
    root = Path(workspace_root) if workspace_root else None
    if root is None:
        return issues
    for item in subtasks:
        if isinstance(item, dict):
            status = str(item.get("status") or "").lower()
            subtask_id = str(item.get("id") or "")
            files_to_create = item.get("files_to_create") or []
        else:
            status = str(getattr(item, "status", "") or "").lower()
            subtask_id = str(getattr(item, "id", "") or "")
            files_to_create = getattr(item, "files_to_create", None) or []
        if status not in {"done", "ok", "complete"}:
            continue
        for path in files_to_create:
            norm = _normalise_candidate_plan_path(str(path))
            if not norm:
                continue
            target = root / norm
            if not _workspace_path_materialized(target):
                issues.append(
                    _issue(
                        "subtask_materialization_missing",
                        f"Subtask `{subtask_id}` is marked done but declared path `{norm}` is missing or empty.",
                        phase=phase,
                        subtask_id=subtask_id,
                    )
                )
    return issues


def validate_completion_contract(
    completion: CompletionContract,
    *,
    context: WorkspaceContext,
    phase: str = "",
) -> list[ContractIssue]:
    completion_context = context
    if completion.changed_files:
        completion_context = replace(
            context,
            current_diff_hash=diff_hash(
                Path(context.workspace_root),
                completion.changed_files,
            ),
        )
    resolver = EvidenceResolver(completion_context)
    issues: list[ContractIssue] = []
    if not completion.subtask_id:
        issues.append(
            _issue(
                "missing_subtask_id",
                "CompletionContract.subtask_id is required.",
                phase=phase,
            )
        )
    if not completion.completed_claims:
        issues.append(
            _issue(
                "claim_without_proof",
                "Completion requires at least one completed claim.",
                phase=phase,
                subtask_id=completion.subtask_id,
            )
        )
    all_refs = list(completion.evidence_refs)
    for claim in completion.completed_claims:
        if not claim.proof_refs:
            issues.append(
                _issue(
                    "claim_without_proof",
                    f"Claim `{claim.claim_id}` has no proof refs.",
                    phase=phase,
                    subtask_id=completion.subtask_id,
                )
            )
        all_refs.extend(claim.proof_refs)
    if completion.verification_report is not None:
        issues.extend(
            resolver.validate_verification_report_ref(
                completion.verification_report,
                phase=phase,
                subtask_id=completion.subtask_id,
                allow_workspace_hash_mismatch_if_diff_matches=True,
            )
        )
    elif not all_refs:
        issues.append(
            _issue(
                "missing_evidence_ref",
                "Completion requires ledger-backed evidence refs.",
                phase=phase,
                subtask_id=completion.subtask_id,
            )
        )
    for ref in all_refs:
        if ref.ref_type not in LEDGER_BACKED_EVIDENCE_REF_TYPES:
            issues.append(
                _issue(
                    "non_ledger_evidence_ref",
                    (
                        "Completion proof refs must point to supervisor ledger-backed "
                        "evidence such as ledger_event, verification_report, test_run, "
                        f"mutation_report, or input_sensitivity_report; got `{ref.ref_type or '<missing>'}`."
                    ),
                    phase=phase,
                    subtask_id=completion.subtask_id,
                )
            )
    issues.extend(
        resolver.validate_refs(tuple(all_refs), phase=phase, subtask_id=completion.subtask_id)
    )
    return issues


def validate_verification_report_ref(
    report: VerificationReportRef,
    *,
    context: WorkspaceContext,
    phase: str = "verify",
) -> list[ContractIssue]:
    return EvidenceResolver(context).validate_verification_report_ref(report, phase=phase)


def validate_risk_profile(
    risk: TaskRiskProfile, proofs: Iterable[ProofSpec]
) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    proof_list = list(proofs)
    kinds = {proof.execution.kind for proof in proof_list}
    if risk.verifier_changed:
        issues.append(
            ContractIssue(
                code="requires_human_checkpoint",
                severity="human_required",
                message="Verifier/policy changes require a human checkpoint.",
            )
        )
    if risk.tests_changed and not any(
        "no_test_tampering" in proof.oracle.required_properties for proof in proof_list
    ):
        issues.append(
            _issue(
                "test_tampering_risk",
                "tests_changed requires no_test_tampering proof property.",
            )
        )
    if (risk.llm_or_prompt_logic or risk.high_stub_risk) and not (
        {"input_sensitivity", "metamorphic", "mutation_smoke"} & kinds
    ):
        issues.append(
            _issue(
                "missing_behavioral_oracle",
                "LLM/high-stub-risk work requires input_sensitivity, metamorphic, or mutation_smoke proof.",
            )
        )
    if risk.web_or_http_runtime and not ({"http_boot", "behavioral_http"} & kinds):
        issues.append(
            _issue(
                "missing_runtime_proof",
                "web_or_http_runtime requires http_boot or behavioral_http proof.",
            )
        )
    return issues


# Planning-phase exits must not re-validate execute/verify completion evidence.
# After loop_back_to(plan), the workspace often grows while the latest
# mark_subtask_complete proof still carries an older workspace_hash.
_PLANNING_EXIT_PHASES = frozenset(
    {"preflight", "research", "research_review", "plan", "plan_review"}
)


class ContractValidator:
    @staticmethod
    def validate(
        bundle: ContractBundle,
        *,
        context: WorkspaceContext | None = None,
        exit_phase: str = "",
        runtime_capabilities: dict[str, bool] | None = None,
        drive_root: Path | None = None,
    ) -> list[ContractIssue]:
        issues: list[ContractIssue] = list(bundle.issues)
        proofs: list[ProofSpec] = []
        resolver = EvidenceResolver(context) if context is not None else None
        skip_cross_phase_evidence = exit_phase in _PLANNING_EXIT_PHASES
        caps = runtime_capabilities or {}
        if not caps and drive_root is not None:
            caps = effective_runtime_capabilities(drive_root)
        from lethyx.contracts.capability_declaration import (
            declaration_ready_for_handoff,
            load_capability_declaration,
        )

        declaration = load_capability_declaration(drive_root)
        declaration_ready = declaration_ready_for_handoff(declaration)
        if bundle.plan is not None:
            if not declaration_ready:
                issues.append(
                    _issue(
                        "missing_capability_declaration",
                        "Plan validation requires a submitted capability_declaration from research.",
                        phase="plan",
                    )
                )
            else:
                issues.extend(validate_plan_capability_consistency(bundle.plan, declaration))
        if bundle.plan is not None:
            issues.extend(validate_plan_candidate_paths(bundle.plan))
            issues.extend(validate_plan_test_tampering_policy(bundle.plan))
            issues.extend(validate_plan_layout_policy(bundle.plan, context=context))
            issues.extend(validate_plan_src_layout_command_proofs(bundle.plan))
            issues.extend(validate_plan_declared_change_scope(bundle.plan))
            issues.extend(
                validate_plan_python_module_entrypoint_scope(bundle.plan, context=context)
            )
            issues.extend(validate_plan_pytest_target_scope(bundle.plan, context=context))
            for subtask in bundle.plan.subtasks:
                if subtask.proof is None:
                    issues.append(
                        _issue(
                            "missing_proof",
                            "Subtask has no typed proof.",
                            subtask_id=subtask.id,
                        )
                    )
                else:
                    proofs.append(subtask.proof)
                    issues.extend(
                        validate_proof_spec(
                            subtask.proof,
                            phase="plan",
                            subtask_id=subtask.id,
                            resolver=resolver,
                            runtime_capabilities=caps,
                            drive_root=drive_root,
                        )
                    )
                    generated_contract = (
                        subtask.generated_test_contract or subtask.proof.generated_test_contract
                    )
                    issues.extend(
                        generated_oracle_contract_issues(
                            generated_contract,
                            subtask_id=subtask.id,
                        )
                    )
                    issues.extend(validate_subtask_proof_strength(subtask, phase="plan"))
        for review in bundle.reviews:
            issues.extend(validate_review_contract(review))
            for item in review.issues:
                issues.append(
                    ContractIssue(
                        code=item.code,
                        severity=item.severity,
                        phase=item.phase,
                        subtask_id=item.subtask_id,
                        message=item.message,
                        evidence_refs=item.evidence_refs,
                        target_subtask_id=item.target_subtask_id,
                        target_path=item.target_path,
                        contract_path=item.contract_path,
                        invalid_values=item.invalid_values,
                        required_deltas=item.required_deltas,
                        failure_hash=item.failure_hash,
                    )
                )
        if context is not None and not skip_cross_phase_evidence:
            for completion in bundle.completions:
                issues.extend(validate_completion_contract(completion, context=context))
            for report in bundle.verification_reports:
                issues.extend(validate_verification_report_ref(report, context=context))
        issues.extend(validate_risk_profile(bundle.risk, proofs))
        return issues
