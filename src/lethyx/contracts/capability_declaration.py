"""Discovery-backed capability declarations for plan gating (direction C)."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lethyx.contracts.models import EvidenceRef, ProofSpec

log = logging.getLogger(__name__)

CAPABILITY_DECLARATION_FILENAME = "capability_declaration.json"
CAPABILITY_DECLARATION_LATEST = "capability_declaration_latest.json"

_CAPABILITY_SOURCES = frozenset({"probe", "declared", "inferred"})
_UNRESOLVED_CAPABILITY_VERIFICATION_PHRASES = (
    "not verified",
    "not yet verified",
    "not proven",
    "not yet proven",
    "unverified",
    "unproven",
    "requires verification",
    "require verification",
    "needs verification",
    "need verification",
    "probe verification",
    "probe confirmation",
    "requires a probe",
    "needs a probe",
    "probe will confirm",
    "will probe",
    "deferred",
)
_CONCRETE_CAPABILITY_LIMITATION_PHRASES = (
    "policy",
    "constraint",
    "forbid",
    "forbidden",
    "denied",
    "unsupported",
    "not supported",
    "not installed",
    "missing",
    "absent",
    "no display access",
    "no display server",
    "without display access",
    "without display",
    "display unset",
    "display missing",
    "headless host",
    "headless ci",
    "headless runner",
    "headless environment",
    "sandbox",
    "platform does not support",
    "failed probe",
    "probe failed",
)
_PREFERENCE_CAPABILITY_UNAVAILABLE_PHRASES = (
    "not needed",
    "not required",
    "not suitable",
    "not appropriate",
    "unnecessary",
    "overkill",
    "prefer",
    "task requires",
    "task needs",
    "will use",
    "instead",
    "not useful",
    "out of scope",
)
_CAPABILITY_TEXT_ALIASES = {
    "desktop_gui_runtime": (
        "desktop_gui_runtime",
        "desktop gui runtime",
        "real-window gui runtime",
        "real window gui runtime",
        "native gui runtime",
        "gui runtime",
        "display access",
    ),
}
_CAPABILITY_TAG_ALIASES = {
    "node_js": "node",
    "nodejs": "node",
    "nodejs_runtime": "node",
    "node_runtime": "node",
    "npm": "node",
    "tsc": "node",
    "typescript": "node",
    "web_server": "http",
    "http_server": "http",
    "local_web_server": "http",
    "localhost_server": "http",
    "web_runtime": "http",
    "fastapi": "http",
    "llm_api": "llm_runtime",
    "llm_provider": "llm_runtime",
    "model_api": "llm_runtime",
}
_DOMAIN_ONLY_CAPABILITY_TAGS = frozenset({"multi_agent_gmas", "gmas"})
# Capabilities that are always available on any host that can already run
# python/subprocess (loopback HTTP server/client). They are not probe-required
# and never need an explicit declaration to satisfy a proof; an explicit
# unavailable declaration still wins.
BASELINE_AVAILABLE_CAPABILITIES = frozenset({"http"})
_HARNESS_PROFILE_ONLY_TAGS = frozenset(
    {
        "web_ui_browser",
        "playwright_e2e",
        "python_src_layout",
        "node_package",
        "cli_app",
        "go_module",
        "rust_crate",
    }
)
_NATIVE_DESKTOP_GUI_INTENT_PHRASES = (
    "tkinter",
    "customtkinter",
    "pyqt",
    "pyside",
    "wxpython",
    "native desktop gui",
    "desktop gui",
    "real-window gui",
    "real window gui",
)
_NATIVE_DESKTOP_GUI_NEGATION_PHRASES = (
    "no native gui",
    "no native desktop gui",
    "no desktop gui",
    "not native gui",
    "not a native gui",
    "not a native desktop gui",
    "not a desktop gui",
    "not needed",
    "not required",
    "not needed",
    "not necessary",
    "unnecessary",
    "not used",
    "no native gui needed",
    "no native desktop gui required",
)


def _text_contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(phrase in lowered for phrase in phrases)


def _alias_windows(text: str, alias: str, *, radius: int = 120) -> list[str]:
    lowered = str(text or "").lower()
    needle = str(alias or "").lower()
    if not needle:
        return []
    windows: list[str] = []
    start = 0
    while True:
        index = lowered.find(needle, start)
        if index < 0:
            return windows
        windows.append(lowered[max(0, index - radius) : index + len(needle) + radius])
        start = index + len(needle)


def _has_positive_native_gui_intent(text: str) -> bool:
    haystack = str(text or "").lower()
    if not haystack.strip():
        return False
    for phrase in _NATIVE_DESKTOP_GUI_INTENT_PHRASES:
        for window in _alias_windows(haystack, phrase, radius=80):
            if _text_contains_any(window, _NATIVE_DESKTOP_GUI_NEGATION_PHRASES):
                continue
            return True
    return False


def _is_capability_slug(name: str) -> bool:
    text = str(name or "").strip()
    if not text or len(text) > 64 or not text[0].islower():
        return False
    for ch in text:
        if ch.islower() or ch.isdigit() or ch in "_-":
            continue
        return False
    return True


@dataclass(frozen=True)
class CapabilityEntry:
    available: bool
    source: str = "probe"
    reason: str = ""
    probe: dict[str, Any] | None = None
    execution_environment_id: str = ""
    python_executable: str = ""
    cwd: str = ""
    env_hash: str = ""

    @classmethod
    def from_mapping(cls, value: Any) -> "CapabilityEntry | None":
        if not isinstance(value, dict):
            return None
        source = str(value.get("source") or "probe").strip().lower()
        if source not in _CAPABILITY_SOURCES:
            source = "declared"
        probe = value.get("probe")
        return cls(
            available=bool(value.get("available")),
            source=source,
            reason=str(value.get("reason") or "").strip(),
            probe=probe if isinstance(probe, dict) else None,
            execution_environment_id=str(
                value.get("execution_environment_id") or value.get("env_id") or ""
            ).strip(),
            python_executable=str(value.get("python_executable") or "").strip(),
            cwd=str(value.get("cwd") or "").strip(),
            env_hash=str(value.get("env_hash") or "").strip(),
        )


@dataclass
class CapabilityDeclaration:
    schema_version: str = "1"
    status: str = "draft"
    run_id: str = ""
    workspace_id: str = ""
    actor: str = "agent"
    capabilities: dict[str, CapabilityEntry] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    notes: str = ""
    evidence_refs: tuple[EvidenceRef, ...] = ()
    discovery_channels: tuple[dict[str, str], ...] = ()
    recommended_skills: tuple[str, ...] = ()
    probe_audit: dict[str, bool] = field(default_factory=dict)
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        caps_out: dict[str, Any] = {}
        for key, entry in self.capabilities.items():
            item: dict[str, Any] = {
                "available": entry.available,
                "source": entry.source,
                "reason": entry.reason,
            }
            if entry.probe:
                item["probe"] = entry.probe
            if entry.execution_environment_id:
                item["execution_environment_id"] = entry.execution_environment_id
            if entry.python_executable:
                item["python_executable"] = entry.python_executable
            if entry.cwd:
                item["cwd"] = entry.cwd
            if entry.env_hash:
                item["env_hash"] = entry.env_hash
            caps_out[key] = item
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "run_id": self.run_id,
            "workspace_id": self.workspace_id,
            "actor": self.actor,
            "capabilities": caps_out,
            "constraints": list(self.constraints),
            "limitations": list(self.limitations),
            "notes": self.notes,
            "evidence_refs": [
                {
                    "ref_type": ref.ref_type,
                    "ref_id": ref.ref_id,
                    "produced_by": ref.produced_by,
                    "hash": ref.hash,
                    "phase": ref.phase,
                    "subtask_id": ref.subtask_id,
                }
                for ref in self.evidence_refs
            ],
            "discovery_channels": [dict(item) for item in self.discovery_channels],
            "recommended_skills": list(self.recommended_skills),
            "probe_audit": dict(self.probe_audit),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "CapabilityDeclaration":
        caps: dict[str, CapabilityEntry] = {}
        raw_caps = value.get("capabilities")
        if isinstance(raw_caps, dict):
            for key, item in raw_caps.items():
                tag = _normalize_capability_tag(str(key)) or ""
                if not tag:
                    continue
                entry = CapabilityEntry.from_mapping(item)
                if entry is not None:
                    caps[tag] = _merge_capability_entries(caps.get(tag), entry)
        refs = value.get("evidence_refs") or ()
        return cls(
            schema_version=str(value.get("schema_version") or "1"),
            status=str(value.get("status") or "draft").strip().lower(),
            run_id=str(value.get("run_id") or "").strip(),
            workspace_id=str(value.get("workspace_id") or "").strip(),
            actor=str(value.get("actor") or "agent").strip(),
            capabilities=caps,
            constraints=tuple(
                str(item).strip() for item in (value.get("constraints") or ()) if str(item).strip()
            ),
            limitations=tuple(
                str(item).strip() for item in (value.get("limitations") or ()) if str(item).strip()
            ),
            notes=str(value.get("notes") or "").strip(),
            evidence_refs=tuple(
                EvidenceRef.from_mapping(item) for item in refs if isinstance(item, dict)
            ),
            discovery_channels=tuple(
                item
                for item in (
                    _normalize_discovery_channel_row(row)
                    for row in (value.get("discovery_channels") or ())
                )
                if item is not None
            ),
            recommended_skills=tuple(
                str(item).strip()
                for item in (value.get("recommended_skills") or ())
                if str(item).strip()
            ),
            probe_audit={
                (_normalize_capability_tag(str(k)) or str(k).strip()): bool(v)
                for k, v in (value.get("probe_audit") or {}).items()
                if isinstance(value.get("probe_audit"), dict)
                and (_normalize_capability_tag(str(k)) or str(k).strip())
            },
            updated_at=float(value.get("updated_at") or 0.0),
        )


def _normalize_discovery_channel_row(row: Any) -> dict[str, str] | None:
    from lethyx.workspace_registry.charter import normalize_discovery_channel

    return normalize_discovery_channel(row)


def _normalize_capability_tag(tag: str) -> str | None:
    name = str(tag or "").strip().lower()
    if not _is_capability_slug(name):
        return None
    return _CAPABILITY_TAG_ALIASES.get(name, name)


def _merge_capability_entries(
    current: CapabilityEntry | None,
    incoming: CapabilityEntry,
) -> CapabilityEntry:
    if current is None:
        return incoming
    if incoming.available != current.available:
        return incoming if incoming.available else current
    if incoming.source == "probe" and current.source != "probe":
        return incoming
    if incoming.probe and not current.probe:
        return incoming
    if incoming.reason and not current.reason:
        return incoming
    return current


def _merge_capability_mapping_entries(
    current: Any,
    incoming: Any,
) -> Any:
    current_entry = CapabilityEntry.from_mapping(
        current if isinstance(current, dict) else {"available": bool(current)}
    )
    incoming_entry = CapabilityEntry.from_mapping(
        incoming if isinstance(incoming, dict) else {"available": bool(incoming)}
    )
    if incoming_entry is None:
        return current
    if current_entry is None:
        return incoming
    merged = _merge_capability_entries(current_entry, incoming_entry)
    return incoming if merged is incoming_entry else current


def normalize_capability_mapping(capabilities: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, item in (capabilities or {}).items():
        tag = _normalize_capability_tag(str(key))
        if not tag:
            continue
        if tag in normalized:
            normalized[tag] = _merge_capability_mapping_entries(normalized[tag], item)
        else:
            normalized[tag] = item
    return normalized


def normalize_probe_audit_mapping(probe_audit: dict[str, Any]) -> dict[str, bool]:
    normalized: dict[str, bool] = {}
    for key, value in (probe_audit or {}).items():
        tag = _normalize_capability_tag(str(key))
        if tag:
            normalized[tag] = bool(value)
    return normalized


def validate_declaration_payload(payload: dict[str, Any]) -> list[str]:
    from lethyx.contracts.harness_profiles import probe_required_capability_ids
    from lethyx.contracts.runtime_probes import validate_probe_spec

    errors: list[str] = []
    if str(payload.get("schema_version") or "1") != "1":
        errors.append("schema_version must be '1'.")
    status = str(payload.get("status") or "").strip().lower()
    if status not in {"draft", "submitted"}:
        errors.append("status must be draft or submitted.")
    caps = payload.get("capabilities")
    if not isinstance(caps, dict) or not caps:
        errors.append("capabilities must be a non-empty object.")
    elif isinstance(caps, dict):
        caps = normalize_capability_mapping(caps)
        probe_audit = normalize_probe_audit_mapping(
            payload.get("probe_audit") if isinstance(payload.get("probe_audit"), dict) else {}
        )
        for key, item in caps.items():
            tag = _normalize_capability_tag(str(key))
            if not tag:
                errors.append(
                    f"invalid capability tag `{key}` (use lowercase slug: "
                    "network, docker, browser_ui, ...)."
                )
                continue
            if isinstance(item, dict) and isinstance(item.get("probe"), dict):
                probe_issue = validate_probe_spec(item["probe"], capability_tag=tag)
                if probe_issue:
                    errors.append(f"capability `{tag}` probe invalid: {probe_issue}")
            elif not isinstance(item, (bool, dict)):
                errors.append(f"capability `{tag}` must be bool or object.")
        for tag in sorted(probe_required_capability_ids()):
            item = caps.get(tag)
            if item is True:
                errors.append(
                    f"capability `{tag}` requires a successful probe under the same "
                    f"capability slug; pass capabilities.{tag}.probe or probes.{tag}, "
                    "or mark it unavailable."
                )
                continue
            if not isinstance(item, dict) or not bool(item.get("available")):
                continue
            source = str(item.get("source") or "").strip().lower()
            if (
                source != "probe"
                or not isinstance(item.get("probe"), dict)
                or probe_audit.get(tag) is not True
            ):
                errors.append(
                    f"capability `{tag}` requires a successful probe under the same "
                    f"capability slug; pass capabilities.{tag}.probe or probes.{tag}, "
                    "or mark it unavailable."
                )
        errors.extend(
            capability_preference_unavailability_errors(
                caps,
                probe_audit=probe_audit,
            )
        )
        errors.extend(
            mentioned_capability_missing_errors(
                caps,
                _capability_payload_text_parts(payload),
            )
        )
        errors.extend(
            probe_required_capability_resolution_errors(
                caps,
                _capability_payload_text_parts(payload),
                probe_audit=probe_audit,
                probe_required=probe_required_capability_ids(),
            )
        )
        errors.extend(
            native_gui_capability_alignment_errors(
                caps,
                _capability_payload_text_parts(payload),
                probe_audit=probe_audit,
            )
        )
    notes = str(payload.get("notes") or "").strip()
    if status == "submitted" and len(notes) < 20:
        errors.append("submitted declaration requires notes (min 20 chars).")
    channels = payload.get("discovery_channels")
    if channels is not None and not isinstance(channels, list):
        errors.append("discovery_channels must be an array.")
    elif isinstance(channels, list):
        for idx, row in enumerate(channels):
            if _normalize_discovery_channel_row(row) is None:
                errors.append(f"discovery_channels[{idx}] is invalid.")
    skills = payload.get("recommended_skills")
    if skills is not None and not isinstance(skills, list):
        errors.append("recommended_skills must be an array.")
    return errors


def _capability_payload_text_parts(payload: dict[str, Any]) -> list[str]:
    text_parts: list[str] = []
    for key in ("notes",):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            text_parts.append(value)
    for key in ("constraints", "limitations"):
        value = payload.get(key)
        if isinstance(value, (list, tuple, set, frozenset)):
            text_parts.extend(str(item) for item in value if str(item).strip())
    return text_parts


def mentioned_capability_missing_errors(
    capabilities: dict[str, Any],
    text_parts: list[str] | tuple[str, ...],
) -> list[str]:
    haystack = "\n".join(text_parts)
    if not haystack.strip():
        return []
    errors: list[str] = []
    for tag, aliases in _CAPABILITY_TEXT_ALIASES.items():
        if tag in capabilities:
            continue
        if any(alias.lower() in haystack.lower() for alias in aliases):
            errors.append(
                f"capability `{tag}` is mentioned in notes/constraints but is "
                f"missing from capabilities.{tag}; add an explicit entry. If "
                "available, it must satisfy the probe contract. If unavailable, "
                "mark available=false with a concrete reason."
            )
    return errors


def capability_preference_unavailability_errors(
    capabilities: dict[str, Any],
    *,
    probe_audit: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    audit = probe_audit if isinstance(probe_audit, dict) else {}
    for tag, item in sorted(capabilities.items()):
        if not isinstance(item, dict) or bool(item.get("available")):
            continue
        reason = str(item.get("reason") or "").strip()
        if not reason or not _text_contains_any(reason, _PREFERENCE_CAPABILITY_UNAVAILABLE_PHRASES):
            continue
        if _text_contains_any(reason, _CONCRETE_CAPABILITY_LIMITATION_PHRASES):
            continue
        if (
            str(item.get("source") or "").strip().lower() == "probe"
            and isinstance(item.get("probe"), dict)
            and audit.get(tag) is False
        ):
            continue
        errors.append(
            f"capability `{tag}` is marked unavailable for task suitability or "
            "planning preference, not because the platform/tool capability is "
            "actually unavailable. Capability declarations describe what can "
            "run; keep preference in notes/plan, or record a failed same-slug "
            "probe/concrete platform limitation."
        )
    return errors


def probe_required_capability_resolution_errors(
    capabilities: dict[str, Any],
    text_parts: list[str] | tuple[str, ...],
    *,
    probe_audit: dict[str, Any] | None = None,
    probe_required: set[str] | frozenset[str] | tuple[str, ...] = (),
) -> list[str]:
    haystack = "\n".join(text_parts)
    errors: list[str] = []
    audit = probe_audit if isinstance(probe_audit, dict) else {}
    for tag in sorted(str(item) for item in probe_required):
        item = capabilities.get(tag)
        if not isinstance(item, dict) or bool(item.get("available")):
            continue
        source = str(item.get("source") or "").strip().lower()
        reason = str(item.get("reason") or "").strip()
        combined = "\n".join(part for part in (reason, haystack) if part)
        if source == "probe" and isinstance(item.get("probe"), dict) and audit.get(tag) is False:
            continue
        if _text_contains_any(combined, _UNRESOLVED_CAPABILITY_VERIFICATION_PHRASES):
            errors.append(
                f"capability `{tag}` is marked unavailable because it still "
                "needs verification. Run a failed same-slug probe "
                f"(capabilities.{tag}.probe or probes.{tag}) before declaring "
                "it unavailable, or record a concrete platform/policy "
                "limitation in constraints/limitations. The probe is part of "
                "the same submit_capability_declaration call; for Tkinter "
                "desktop runtime use e.g. "
                'probes.desktop_gui_runtime={"kind":"command","intent":"real_gui_root_lifecycle","command":["python","-c","import tkinter as tk; root=tk.Tk(); root.update(); root.destroy()"],"expect_exit":0}.'
            )
            continue
        if not _text_contains_any(combined, _CONCRETE_CAPABILITY_LIMITATION_PHRASES):
            errors.append(
                f"capability `{tag}` is marked unavailable without a failed "
                "same-slug probe or concrete platform/policy limitation. "
                f"Attach capabilities.{tag}.probe/probes.{tag} in this "
                "submit_capability_declaration call, or explain the hard "
                "limitation in constraints/limitations."
            )
    return errors


def native_gui_capability_alignment_errors(
    capabilities: dict[str, Any],
    text_parts: list[str] | tuple[str, ...],
    *,
    probe_audit: dict[str, Any] | None = None,
) -> list[str]:
    combined_parts = list(text_parts)
    for tag in ("desktop_gui_headless", "desktop_gui_runtime"):
        item = capabilities.get(tag)
        if isinstance(item, dict) and str(item.get("reason") or "").strip():
            combined_parts.append(str(item.get("reason") or ""))
    haystack = "\n".join(combined_parts)
    if not _has_positive_native_gui_intent(haystack):
        return []

    headless = capabilities.get("desktop_gui_headless")
    if isinstance(headless, dict) and bool(headless.get("available")):
        return []

    runtime = capabilities.get("desktop_gui_runtime")
    audit = probe_audit if isinstance(probe_audit, dict) else {}
    if (
        isinstance(runtime, dict)
        and bool(runtime.get("available"))
        and str(runtime.get("source") or "").strip().lower() == "probe"
        and isinstance(runtime.get("probe"), dict)
        and audit.get("desktop_gui_runtime") is True
    ):
        return []

    return [
        "native desktop GUI handoff mentions Tkinter/PyQt/PySide/wxPython "
        "or a desktop GUI, but capability_declaration does not expose a usable "
        "GUI harness capability. Declare desktop_gui_headless available for "
        "headless adapter/controller proof, or desktop_gui_runtime available "
        "with a same-slug probe for real-window proof. If neither can run, "
        "record a concrete blocker/limitation instead of recommending native GUI."
    ]


def validate_discovery_coverage(
    declaration: CapabilityDeclaration | None,
    *,
    charter: dict[str, Any],
    allowed_tools: set[str] | frozenset[str],
    research_depth: str,
) -> str | None:
    from lethyx.workspace_registry.charter import charter_required_discovery_tools

    if str(research_depth or "").strip().lower() != "full":
        return None
    if declaration is None:
        return "missing capability_declaration for full research depth."
    declared = {
        str(item.get("tool") or ""): str(item.get("outcome") or "")
        for item in declaration.discovery_channels
    }
    required_tools = charter_required_discovery_tools(charter, allowed_tools=allowed_tools)
    if required_tools:
        missing = [tool for tool in required_tools if tool not in declared]
        if missing:
            return (
                "discovery coverage incomplete for charter-required channels: "
                + ", ".join(missing)
                + ". Record each in capability_declaration.discovery_channels "
                "(discovery coverage)."
            )
        return None
    if len(declared) < 2:
        return (
            "discovery coverage: research_depth=full requires discovery_channels "
            "in capability_declaration (min 2 tools with outcome), or charter "
            "[[discovery.required_channels]] in workspace.toml."
        )
    return None


def declaration_effective_capabilities(
    declaration: CapabilityDeclaration | None,
    *,
    probed: dict[str, bool] | None = None,
) -> dict[str, bool]:
    from lethyx.contracts.harness_profiles import probe_required_capability_ids

    if declaration is None:
        return dict(probed or {})
    probe_required = probe_required_capability_ids()
    effective: dict[str, bool] = {}
    for tag, entry in declaration.capabilities.items():
        normalized_tag = _normalize_capability_tag(tag)
        if not normalized_tag:
            continue
        if (
            normalized_tag in probe_required
            and entry.available
            and (
                entry.source != "probe"
                or entry.probe is None
                or declaration.probe_audit.get(normalized_tag) is not True
            )
        ):
            effective[normalized_tag] = False
            continue
        if entry.available:
            effective[normalized_tag] = True
        else:
            effective.setdefault(normalized_tag, False)
    return effective


def declaration_ready_for_handoff(declaration: CapabilityDeclaration | None) -> bool:
    if declaration is None:
        return False
    if declaration.status != "submitted":
        return False
    if not declaration.capabilities:
        return False
    return len(declaration.notes.strip()) >= 20


def _proof_explicit_required_capabilities(proof: ProofSpec | dict[str, Any]) -> tuple[str, ...]:
    if isinstance(proof, ProofSpec):
        raw = proof.required_capabilities
    else:
        raw = proof.get("required_capabilities") if isinstance(proof, dict) else None
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(
        tag
        for tag in (_normalize_capability_tag(str(item)) for item in raw)
        if tag and tag not in _HARNESS_PROFILE_ONLY_TAGS
    )


def proof_required_capabilities(proof: ProofSpec | dict[str, Any]) -> frozenset[str]:
    explicit = _proof_explicit_required_capabilities(proof)
    if explicit:
        return frozenset(explicit)
    if isinstance(proof, ProofSpec):
        kind = proof.execution.kind
        command = proof.execution.command
    else:
        execution = proof.get("execution") if isinstance(proof.get("execution"), dict) else {}
        kind = str(execution.get("kind") or "")
        command = execution.get("command") or ()
    required: set[str] = set()
    if kind and kind != "none" and command:
        required.add("subprocess")
    return frozenset(required)


def validate_proof_against_capabilities(
    proof: ProofSpec | dict[str, Any],
    effective_caps: dict[str, bool],
    *,
    subtask_id: str = "",
) -> str | None:
    missing: list[str] = []
    undeclared: list[str] = []
    domain_tags: list[str] = []
    for tag in sorted(proof_required_capabilities(proof)):
        if tag in _DOMAIN_ONLY_CAPABILITY_TAGS:
            domain_tags.append(tag)
            continue
        if tag not in effective_caps:
            # Universally-available, non-probe-required capabilities (e.g. a
            # localhost HTTP server/client) do not need an explicit declaration:
            # any host that can run python/subprocess can bind a loopback socket.
            # They only block when the declaration explicitly marks them False.
            if tag in BASELINE_AVAILABLE_CAPABILITIES:
                continue
            undeclared.append(tag)
        elif effective_caps[tag] is False:
            missing.append(tag)
    if not missing and not undeclared and not domain_tags:
        return None
    suffix = f" (subtask `{subtask_id}`)" if subtask_id else ""
    parts: list[str] = []
    if domain_tags:
        parts.append(
            "domain_not_runtime_capability: "
            + ", ".join(domain_tags)
            + " belong in proof.domain/subtask.domain or memory_scope assets, not proof.required_capabilities"
        )
    if undeclared:
        parts.append("undeclared in capability_declaration: " + ", ".join(undeclared))
    if missing:
        parts.append("marked unavailable: " + ", ".join(missing))
    guidance = (
        ". Add explicit proof.required_capabilities and matching "
        "capability_declaration entries (with probe commands when needed)."
    )
    optional_driver_caps = {"pyautogui", "selenium", "playwright"}
    if any(tag in optional_driver_caps for tag in undeclared):
        guidance += (
            " If this happens in plan phase and the driver-only capability is "
            "not essential to the task, remove optional driver-only capabilities "
            "from the proof and prefer an already declared proof shape such as "
            "desktop_gui_headless/controller assertions. If the capability is "
            "genuinely required, loop back to research for a capability "
            "declaration update before submitting the plan."
        )
    return "Proof requires capabilities not satisfied: " + "; ".join(parts) + suffix + guidance


def _declaration_paths(drive_root: Path) -> tuple[Path, Path]:
    state = drive_root / "state"
    return (
        state / CAPABILITY_DECLARATION_FILENAME,
        state / CAPABILITY_DECLARATION_LATEST,
    )


def load_capability_declaration(drive_root: Path | None) -> CapabilityDeclaration | None:
    if drive_root is None:
        return None
    for path in _declaration_paths(drive_root):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.debug("capability declaration read failed: %s", exc, exc_info=True)
            continue
        if isinstance(payload, dict):
            return CapabilityDeclaration.from_mapping(payload)
    return None


def persist_capability_declaration(
    drive_root: Path,
    payload: dict[str, Any],
) -> Path:
    state = drive_root / "state"
    state.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("schema_version", "1")
    if isinstance(payload.get("capabilities"), dict):
        payload["capabilities"] = normalize_capability_mapping(payload["capabilities"])
    if isinstance(payload.get("probe_audit"), dict):
        payload["probe_audit"] = normalize_probe_audit_mapping(payload["probe_audit"])
    payload["updated_at"] = time.time()
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    primary, latest = _declaration_paths(drive_root)
    primary.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    try:
        from lethyx.contracts.runtime_probes import persist_runtime_capabilities

        declaration = CapabilityDeclaration.from_mapping(payload)
        persist_runtime_capabilities(
            drive_root,
            declaration_effective_capabilities(declaration),
            source="capability_declaration",
        )
    except Exception:
        log.debug(
            "failed to mirror capability declaration into runtime audit",
            exc_info=True,
        )
    return primary


def build_declaration_from_probes(
    *,
    run_id: str,
    workspace_id: str,
    probed: dict[str, bool],
    actor: str = "harness",
    status: str = "draft",
    notes: str = "",
    constraints: list[str] | None = None,
) -> dict[str, Any]:
    capabilities = {
        tag: {
            "available": bool(value),
            "source": "probe",
            "reason": "" if value else "baseline probe reported unavailable",
        }
        for tag, value in sorted(probed.items())
    }
    return {
        "schema_version": "1",
        "status": status,
        "run_id": run_id,
        "workspace_id": workspace_id,
        "actor": actor,
        "capabilities": capabilities,
        "constraints": constraints or [],
        "limitations": [],
        "notes": notes,
        "evidence_refs": [],
        "discovery_channels": [],
        "recommended_skills": [],
        "probe_audit": dict(probed),
        "updated_at": time.time(),
    }


def ensure_probe_backed_declaration(
    drive_root: Path,
    workspace_root: Path,
    *,
    run_id: str,
    workspace_id: str,
    actor: str = "harness",
) -> CapabilityDeclaration:
    from lethyx.contracts.runtime_probes import (
        baseline_runtime_capabilities,
        persist_runtime_capabilities,
    )

    _ = workspace_root
    existing = load_capability_declaration(drive_root)
    if existing is not None and existing.capabilities:
        return existing
    probed = baseline_runtime_capabilities()
    persist_runtime_capabilities(drive_root, probed)
    payload = build_declaration_from_probes(
        run_id=run_id,
        workspace_id=workspace_id,
        probed=probed,
        actor=actor,
        status="draft",
        notes=(
            "Draft baseline only (python/subprocess). Research must call "
            "submit_capability_declaration with task-specific capabilities "
            "and optional probe commands before handoff."
        ),
    )
    persist_capability_declaration(drive_root, payload)
    return CapabilityDeclaration.from_mapping(payload)
