"""Typed harness profile catalog for phase context and enforcement.

Harness profiles are small orchestration contracts. They are not skills and not
domain memory: planning sees a compact catalog, execution receives only the
profiles selected for the active subtask, and guards consume validator flags.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


HARNESS_PROFILE_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class HarnessProfile:
    id: str
    title: str
    summary: str
    canonical_proof_template: str = ""
    applies_to: tuple[str, ...] = ()
    planner_contract: tuple[str, ...] = ()
    execute_contract: tuple[str, ...] = ()
    validator_flags: tuple[str, ...] = ()
    memory_scope_hints: tuple[str, ...] = ()
    probe_required_capabilities: tuple[str, ...] = ()

    def catalog_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "canonical_proof_template": self.canonical_proof_template,
            "applies_to": list(self.applies_to),
            "probe_required_capabilities": list(self.probe_required_capabilities),
        }

    def contract_payload(self, *, reason: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "canonical_proof_template": self.canonical_proof_template,
            "reason": reason,
            "planner_contract": list(self.planner_contract),
            "execute_contract": list(self.execute_contract),
            "validator_flags": list(self.validator_flags),
            "memory_scope_hints": list(self.memory_scope_hints),
            "probe_required_capabilities": list(self.probe_required_capabilities),
        }


_PROFILES: dict[str, HarnessProfile] = {
    "python_src_layout": HarnessProfile(
        id="python_src_layout",
        title="Python src layout",
        canonical_proof_template="python_src_unit",
        summary=(
            "Greenfield Python application/library work uses one canonical "
            "src/<package>/ package root, tests under tests/, and package "
            "entrypoints under the package rather than root ad-hoc scripts."
        ),
        applies_to=("greenfield Python packages", "Python CLI/apps/libraries"),
        planner_contract=(
            "Declare production modules under exactly one src/<package>/ root.",
            "Use src/<package>/cli.py or src/<package>/__main__.py plus project scripts for launch entrypoints.",
            "Keep tests under tests/ and make proofs runnable from the workspace root.",
        ),
        execute_contract=(
            "If an accepted path conflicts with src layout policy, mutate the subtask scope instead of retrying the same write.",
            "Do not add one-off diagnostic Python scripts to the workspace root, docs/, or src/scripts/.",
        ),
        validator_flags=("python_src_layout",),
        memory_scope_hints=("load package layout notes before editing package roots",),
    ),
    "desktop_gui_headless": HarnessProfile(
        id="desktop_gui_headless",
        title="Desktop GUI headless proof",
        canonical_proof_template="desktop_gui_headless_model",
        summary=(
            "Native desktop GUI work is proven through headless behavioral "
            "tests of state, callbacks, labels, and launch wiring; real display "
            "roots are reserved for optional smoke/e2e when runtime supports it."
        ),
        applies_to=("tkinter", "PyQt/PySide", "wxPython", "native desktop GUI"),
        planner_contract=(
            "Set proof.harness_profile to desktop_gui_headless for native GUI unit-proof leaves.",
            "Plan a testable adapter/model/controller boundary for labels, callbacks, state transitions, and invalid-input behavior.",
            "Use real-display launch only as a separate smoke/e2e check when capability probes show it is available.",
        ),
        execute_contract=(
            "Keep unit proof headless: assert behavior through adapter/model/controller APIs.",
            "Do not create a real native GUI root inside proof tests; repair production design instead of skipping tests.",
            "If display launch is needed, make it a separate runtime smoke proof gated by capability.",
            "Even headless GUI tests that instantiate toolkit widgets are auto-run under a virtual display (Xvfb) on headless hosts; you do not need to manage $DISPLAY yourself.",
        ),
        validator_flags=("no_native_gui_root_in_unit_proof",),
        memory_scope_hints=("prefer local GUI architecture notes and controller contracts",),
    ),
    "desktop_gui_runtime": HarnessProfile(
        id="desktop_gui_runtime",
        title="Desktop GUI runtime proof",
        canonical_proof_template="desktop_gui_runtime_smoke",
        summary=(
            "Native desktop GUI smoke/e2e work launches the real application "
            "under an explicit display/automation capability, exercises visible "
            "UI behavior, captures command/log/screenshot evidence, and cleans "
            "up the process."
        ),
        applies_to=(
            "tkinter real-window smoke",
            "PyQt/PySide real-window smoke",
            "wxPython real-window smoke",
            "native desktop GUI automation",
        ),
        planner_contract=(
            "Set proof.harness_profile to desktop_gui_runtime only when capability_declaration marks desktop_gui_runtime available.",
            "Include proof.required_capabilities with desktop_gui_runtime plus the language/runtime capabilities needed by the proof command.",
            "Use proof.execution.command as the managed launch command; put structured readiness, optional assert/interaction/driver command, evidence expectations, timeout, and cleanup notes in proof.harness_options.",
            "Keep runtime smoke/e2e separate from headless controller/model unit-proof leaves.",
        ),
        execute_contract=(
            "Real native GUI roots are allowed only inside this runtime smoke/e2e proof path.",
            "Launch the app through the accepted proof command, wait for readiness, perform visible interactions, and assert observable UI/output behavior.",
            "Capture machine-readable evidence such as command output, screenshot path, event log, or toolkit state; clean up windows/processes on timeout.",
            'On headless hosts the probe and the managed launch command are auto-run under a virtual display (Xvfb); do not hand-manage $DISPLAY. If xvfb-run is missing, install it once with provision_workspace_environment {"kind":"apt_install","packages":["xvfb"]}.',
            "If a display truly cannot be provisioned, mutate the plan to a headless GUI proof or rebuild the UI as a localhost web app proven with an http_boot/behavioral_http browser proof instead of faking the runtime.",
        ),
        validator_flags=("native_gui_runtime_proof",),
        memory_scope_hints=(
            "preload GUI runtime/tooling notes and active harness_options before writing runtime proof",
            "load any skill or MCP refs declared by the planner for desktop automation",
        ),
        probe_required_capabilities=("desktop_gui_runtime",),
    ),
    "web_ui_browser": HarnessProfile(
        id="web_ui_browser",
        title="Web UI browser proof",
        summary=(
            "Web UI work uses bootable local runtime proof and browser/http "
            "checks for visible behavior, navigation, and interactions."
        ),
        applies_to=("localhost web apps", "browser UI", "HTTP UI flows"),
        planner_contract=(
            "Use http_boot or behavioral_http/browser proof for user-facing web work.",
            "Declare server start, readiness, and interaction targets explicitly.",
        ),
        execute_contract=(
            "Prefer the browser/runtime proof path over import-only or DOM-free claims.",
            "Keep screenshots/logs/evidence tied to the active subtask proof.",
        ),
        validator_flags=("web_ui_browser_runtime",),
    ),
    "llm_runtime": HarnessProfile(
        id="llm_runtime",
        title="LLM runtime contract",
        summary=(
            "LLM/agent behavior uses the probed inherited runtime and retrieved "
            "API facts; it must not silently replace model behavior with mocks, "
            "static choices, or hardcoded fallbacks."
        ),
        applies_to=("LLM agents", "judges", "GMAS", "prompt/runtime behavior"),
        planner_contract=(
            "Declare required runtime capabilities and behavioral proof for LLM/agent leaves.",
            "Use retrieval-backed API symbols rather than guessed imports or constructors.",
        ),
        execute_contract=(
            "Call the GMAS/LLM knowledge tools before the first write on matching subtasks.",
            "Use public runtime aliases and fail explicitly when required runtime is absent.",
        ),
        validator_flags=("llm_runtime_real_capability",),
        memory_scope_hints=("preload matching GMAS/API references when available",),
    ),
    "cli_app": HarnessProfile(
        id="cli_app",
        title="CLI application proof",
        summary=(
            "Command-line apps expose a package entrypoint/script and prove "
            "observable command behavior through argv-level tests."
        ),
        applies_to=("CLI apps", "console scripts", "package launchers"),
        planner_contract=(
            "Prefer project scripts or python -m <package> over root main.py in greenfield src layouts.",
            "Use behavioral CLI tests that assert stdout/stderr/exit behavior.",
        ),
        execute_contract=(
            "Keep launch wiring thin and test command behavior through stable argv entrypoints.",
            "Do not close CLI leaves with import-only proof.",
        ),
        validator_flags=("cli_behavioral_entrypoint",),
    ),
    "node_package": HarnessProfile(
        id="node_package",
        title="Script-managed web/package proof",
        summary="Package work uses project-declared scripts and typed test targets without Python src-layout assumptions.",
        applies_to=("package manifest", "script-managed package", "typed UI test targets"),
        planner_contract=(
            "Declare package manifest/lockfile changes and use project-declared scripts for build/test proof.",
            "Use proof.scope.test_targets for package/spec files; pytest_targets are Python-only.",
        ),
        execute_contract=(
            "Use provision_workspace_environment for the manifest/lockfile-selected dependency setup operation before proof.",
            "Do not add PYTHONPATH=src for package import/module failures.",
        ),
        validator_flags=("node_package",),
    ),
    "go_module": HarnessProfile(
        id="go_module",
        title="Go module proof",
        summary="Go modules use go.mod/go.sum and package-level behavioral proofs.",
        applies_to=("go.mod", "Go module", "Go CLI/services"),
        planner_contract=(
            "Use module/package-specific proof commands that exercise behavior.",
            "Use proof.scope.test_targets for *_test.go files or package targets.",
        ),
        execute_contract=(
            "Use provision_workspace_environment with go_mod_download for dependency setup.",
            "Keep generated tests in *_test.go files tied to typed oracle claims.",
        ),
        validator_flags=("go_module",),
    ),
    "rust_crate": HarnessProfile(
        id="rust_crate",
        title="Rust crate proof",
        summary="Rust crates use Cargo.toml/Cargo.lock and crate-level behavioral proofs.",
        applies_to=("Cargo.toml", "Rust crate", "Rust CLI/libraries"),
        planner_contract=(
            "Use crate/module-specific proof commands that exercise behavior.",
            "Use proof.scope.test_targets for tests/*.rs files or named Rust tests.",
        ),
        execute_contract=(
            "Use provision_workspace_environment with cargo_fetch for dependency setup.",
            "Do not route Rust build failures through Python import recovery.",
        ),
        validator_flags=("rust_crate",),
    ),
    "playwright_e2e": HarnessProfile(
        id="playwright_e2e",
        title="Playwright e2e proof",
        summary="Browser end-to-end work launches a local app, waits for readiness, runs Playwright, captures artifacts, and cleans up.",
        applies_to=("Playwright", "browser e2e", "web UI runtime"),
        planner_contract=(
            "Represent server launch/readiness and Playwright test command in proof.harness_options.",
            "Use proof.scope.test_targets for e2e spec files and avoid manual-only visual claims.",
        ),
        execute_contract=(
            "Use managed runtime proof with readiness, Playwright command, traces/screenshots when useful, and cleanup.",
            "Keep browser e2e evidence ledger-backed through run_subtask_proof.",
        ),
        validator_flags=("playwright_e2e", "web_ui_browser_runtime"),
    ),
}


def all_harness_profiles() -> tuple[HarnessProfile, ...]:
    return tuple(_PROFILES.values())


def get_harness_profile(profile_id: str) -> HarnessProfile | None:
    return _PROFILES.get(str(profile_id or "").strip())


def known_harness_profile_ids() -> tuple[str, ...]:
    return tuple(_PROFILES.keys())


def validator_flags_for_profile(profile_id: str) -> frozenset[str]:
    profile = get_harness_profile(profile_id)
    if profile is None:
        return frozenset()
    return frozenset(profile.validator_flags)


def probe_required_capability_ids() -> frozenset[str]:
    required: set[str] = set()
    for profile in all_harness_profiles():
        required.update(profile.probe_required_capabilities)
    return frozenset(required)


def _norm_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip().lstrip("/")


def _iter_path_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        text = _norm_path(value)
        if text:
            yield text
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        for item in value:
            text = _norm_path(item)
            if text:
                yield text


def _subtask_paths(subtask: dict[str, Any] | None) -> list[str]:
    if not isinstance(subtask, dict):
        return []
    paths: list[str] = []
    for key in ("files_to_create", "files_to_change", "files_affected"):
        paths.extend(_iter_path_values(subtask.get(key)))
    proof = subtask.get("proof")
    scope = proof.get("scope") if isinstance(proof, dict) else None
    if isinstance(scope, dict):
        for key in (
            "files_under_test",
            "changed_files_expected",
            "pytest_targets",
            "test_targets",
        ):
            paths.extend(_iter_path_values(scope.get(key)))
    return paths


def _proof_payload(subtask: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(subtask, dict):
        return {}
    proof = subtask.get("proof")
    return proof if isinstance(proof, dict) else {}


def explicit_harness_ids_from_subtask(subtask: dict[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(subtask, dict):
        return ()
    raw_values: list[Any] = [
        subtask.get("harness_profile"),
        subtask.get("harness_profiles"),
        subtask.get("harness_id"),
    ]
    harness = subtask.get("harness")
    if isinstance(harness, dict):
        raw_values.append(harness.get("id"))
    proof = _proof_payload(subtask)
    raw_values.extend(
        [
            proof.get("harness_profile"),
            proof.get("harness_profiles"),
            proof.get("harness_id"),
        ]
    )
    proof_harness = proof.get("harness")
    if isinstance(proof_harness, dict):
        raw_values.append(proof_harness.get("id"))
    ids: list[str] = []
    for raw in raw_values:
        candidates = raw if isinstance(raw, list) else [raw]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text and text in _PROFILES and text not in ids:
                ids.append(text)
    return tuple(ids)


def _safe_default_harness_ids_for_subtask(
    subtask: dict[str, Any] | None,
) -> tuple[str, ...]:
    if not isinstance(subtask, dict):
        return ()
    ids: list[str] = []
    paths = _subtask_paths(subtask)
    if any(path.endswith(".py") for path in paths) or "pyproject.toml" in paths:
        ids.append("python_src_layout")
    if any(
        path.endswith((".js", ".jsx", ".ts", ".tsx")) or path == "package.json" for path in paths
    ):
        ids.append("node_package")
    if any(path.endswith(".go") or path == "go.mod" for path in paths):
        ids.append("go_module")
    if any(path.endswith(".rs") or path == "Cargo.toml" for path in paths):
        ids.append("rust_crate")
    return tuple(ids)


def active_harness_ids_for_subtask(
    subtask: dict[str, Any] | None,
) -> tuple[str, ...]:
    ids: list[str] = []
    for profile_id in explicit_harness_ids_from_subtask(subtask):
        if profile_id not in ids:
            ids.append(profile_id)
    for profile_id in _safe_default_harness_ids_for_subtask(subtask):
        if profile_id not in ids:
            ids.append(profile_id)
    return tuple(ids)


def suggest_harness_ids_for_planning(
    subtask: dict[str, Any] | None,
    *,
    phase_id: str = "",
    capability_envelope: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    if not isinstance(subtask, dict):
        return ()
    ids = list(explicit_harness_ids_from_subtask(subtask))
    paths = _subtask_paths(subtask)
    proof = _proof_payload(subtask)
    execution = proof.get("execution") if isinstance(proof.get("execution"), dict) else {}
    execution_kind = str(execution.get("kind") or "").strip().lower()
    required_cap_ids = {
        str(item).strip().lower()
        for item in (proof.get("required_capabilities") or [])
        if str(item).strip()
    }
    harness_options = proof.get("harness_options")
    harness_option_keys = (
        {str(key).strip().lower() for key in (harness_options or {}) if str(key).strip()}
        if isinstance(harness_options, dict)
        else set()
    )

    def add(profile_id: str) -> None:
        if profile_id not in ids and profile_id in _PROFILES:
            ids.append(profile_id)

    if any(path.endswith(".py") for path in paths) or "pyproject.toml" in paths:
        add("python_src_layout")
    if any(
        path.endswith((".js", ".jsx", ".ts", ".tsx")) or path == "package.json" for path in paths
    ):
        add("node_package")
    if any(path.endswith(".go") or path == "go.mod" for path in paths):
        add("go_module")
    if any(path.endswith(".rs") or path == "Cargo.toml" for path in paths):
        add("rust_crate")

    runtime_gui_capabilities = {
        "desktop_gui_runtime",
        "native_gui_runtime",
        "gui_runtime",
        "display_server",
    }
    if execution_kind in runtime_gui_capabilities or required_cap_ids & runtime_gui_capabilities:
        add("desktop_gui_runtime")
    elif execution_kind in {"desktop_gui_headless", "native_gui_headless"}:
        add("desktop_gui_headless")

    web_runtime_kinds = {"http_boot", "behavioral_http", "browser", "browser_automation"}
    web_runtime_capabilities = {"http", "web_ui", "browser", "browser_automation"}
    web_runtime_option_keys = {
        "health_url",
        "readiness_url",
        "browser_command",
        "browser_driver",
        "e2e_command",
    }
    if (
        execution_kind in web_runtime_kinds
        or required_cap_ids & web_runtime_capabilities
        or harness_option_keys & web_runtime_option_keys
    ):
        if execution_kind in {"browser", "browser_automation"} or harness_option_keys & {
            "browser_command",
            "browser_driver",
            "e2e_command",
        }:
            add("playwright_e2e")
        add("web_ui_browser")

    if required_cap_ids & {"llm_api", "llm_runtime", "model_api"}:
        add("llm_runtime")
    if execution_kind in {"cli", "cli_app", "command_line"}:
        add("cli_app")

    caps = capability_envelope or {}
    if isinstance(caps, dict):
        if _capability_envelope_includes_model_runtime(caps):
            add("llm_runtime")
    return tuple(ids[:3])


def infer_harness_ids_for_subtask(
    subtask: dict[str, Any] | None,
    *,
    phase_id: str = "",
    capability_envelope: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Backward-compatible planner suggestion helper.

    This is intentionally not an execute/control-plane authority. Execute
    contexts and validator flags use explicit proof.harness_profile plus safe
    structural defaults only.
    """

    return suggest_harness_ids_for_planning(
        subtask,
        phase_id=phase_id,
        capability_envelope=capability_envelope,
    )


def _capability_entry_available(value: Any) -> bool:
    if isinstance(value, dict):
        available = value.get("available")
        if available is True:
            return True
        if str(available).strip().lower() in {"true", "yes", "1"}:
            return True
        return False
    return value is True or str(value).strip().lower() in {"true", "yes", "1"}


def _capability_envelope_includes_model_runtime(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).strip().lower()
            if key_text in {"llm_api", "llm_runtime", "model_api"}:
                if _capability_entry_available(item):
                    return True
                continue
            if _capability_envelope_includes_model_runtime(item):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_capability_envelope_includes_model_runtime(item) for item in value)
    text = str(value or "").strip().lower()
    return text in {"llm_api", "llm_runtime", "model_api"}


def build_harness_contract_payload(
    *,
    phase_id: str,
    active_subtask: dict[str, Any] | None = None,
    capability_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    phase = str(phase_id or "")
    if phase in {"plan", "plan_review"}:
        caps = capability_envelope if isinstance(capability_envelope, dict) else {}
        include_model_runtime = _capability_envelope_includes_model_runtime(caps)
        profiles = [
            profile.catalog_payload()
            for profile in all_harness_profiles()
            if include_model_runtime or profile.id != "llm_runtime"
        ]
        return {
            "schema_version": HARNESS_PROFILE_SCHEMA_VERSION,
            "mode": "catalog",
            "selected_ids": [],
            "reason": "planning phases receive compact selectable profile catalog",
            "profiles": profiles,
        }
    if phase == "execute" and active_subtask:
        selected = active_harness_ids_for_subtask(active_subtask)
        return {
            "schema_version": HARNESS_PROFILE_SCHEMA_VERSION,
            "mode": "active",
            "selected_ids": list(selected),
            "reason": (
                "active execute subtask explicit profile selection plus safe structural defaults"
            ),
            "profiles": [
                _PROFILES[profile_id].contract_payload(reason="selected for active subtask")
                for profile_id in selected
                if profile_id in _PROFILES
            ],
        }
    return {
        "schema_version": HARNESS_PROFILE_SCHEMA_VERSION,
        "mode": "none",
        "selected_ids": [],
        "reason": "no harness profile applies to this phase context",
        "profiles": [],
    }


def render_harness_contract_markdown(payload: dict[str, Any]) -> str:
    mode = str(payload.get("mode") or "none")
    profiles = payload.get("profiles") if isinstance(payload.get("profiles"), list) else []
    if not profiles or mode == "none":
        return ""
    lines: list[str] = []
    if mode == "catalog":
        lines.append("Planner harness profile catalog:")
        lines.append(
            "Set `proof.harness_profile` to one matching id when a subtask needs profile-specific tools, memory, or guards. Do not apply inactive profiles."
        )
        for profile in profiles:
            suffix = ""
            probe_required = profile.get("probe_required_capabilities") or []
            if probe_required:
                suffix = (
                    " Probe-required capabilities: "
                    + ", ".join(f"`{item}`" for item in probe_required)
                    + "."
                )
            lines.append(f"- `{profile.get('id')}`: {profile.get('summary')}{suffix}")
        return "\n".join(lines)
    lines.append("Active harness contract:")
    lines.append(
        "Lethyx selected these profiles for the current subtask. They guide proof shape and validator flags, but the agent still decides the implementation."
    )
    for profile in profiles:
        lines.append(f"### {profile.get('id')}: {profile.get('title')}")
        summary = str(profile.get("summary") or "").strip()
        if summary:
            lines.append(summary)
        execute_contract = profile.get("execute_contract") or []
        for item in execute_contract[:4]:
            lines.append(f"- {item}")
        flags = profile.get("validator_flags") or []
        if flags:
            lines.append("Validator flags: " + ", ".join(f"`{flag}`" for flag in flags))
        probe_required = profile.get("probe_required_capabilities") or []
        if probe_required:
            lines.append(
                "Probe-required capabilities: " + ", ".join(f"`{item}`" for item in probe_required)
            )
        memory_hints = profile.get("memory_scope_hints") or []
        if memory_hints:
            lines.append("Memory hints: " + "; ".join(str(item) for item in memory_hints[:2]))
    return "\n".join(lines)


def validator_flags_from_harness_payload(payload: dict[str, Any] | None) -> frozenset[str]:
    if not isinstance(payload, dict):
        return frozenset()
    flags: set[str] = set()
    for profile in payload.get("profiles") or []:
        if not isinstance(profile, dict):
            continue
        for flag in profile.get("validator_flags") or []:
            text = str(flag or "").strip()
            if text:
                flags.add(text)
    return frozenset(flags)


def validator_flags_from_overlays(overlays: dict[str, Any] | None) -> frozenset[str]:
    if not isinstance(overlays, dict):
        return frozenset()
    direct = overlays.get("harness_contract")
    if isinstance(direct, dict):
        flags = validator_flags_from_harness_payload(direct)
        if flags:
            return flags
    bundle = overlays.get("llm_input_bundle")
    if isinstance(bundle, dict):
        return validator_flags_from_harness_payload(bundle.get("harness_contract"))
    return frozenset()


def validator_flags_for_subtask(subtask: dict[str, Any] | None) -> frozenset[str]:
    flags: set[str] = set()
    for profile_id in active_harness_ids_for_subtask(subtask):
        flags.update(validator_flags_for_profile(profile_id))
    return frozenset(flags)
