"""Persistent, content-free policy-bundle pins for the local service."""

import hashlib
import json
import math
import os
import re
import tempfile
import threading
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from um_mem.memory.blocks.models import MemoryBlock
from um_mem.memory.paths import normalize_workspace_id

_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,255}$")
_SHA256_IDENTIFIER = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTENT_KEYS = frozenset(
    {
        "content",
        "message",
        "messages",
        "prompt",
        "raw_input",
        "raw_output",
        "text",
        "transcript",
    }
)
_BASELINE_ACTIVATION_REVISION = (
    "activation:um_mem.memory.selection.MemorySelectionPolicy.eligible_for_overlay@1"
)
_BASELINE_WORKING_CONTEXT_REVISION = (
    "working_context:um_mem.memory.working_continuity.compile_continuation_packet@1"
)
_BASELINE_RECALL_OPERATOR_REVISION = (
    "recall_operator:um_mem.memory.recall_plan.RecallPlan.compile@2"
)
_BASELINE_CONSOLIDATION_REVISION = (
    "consolidation:um_mem.memory.consolidation.CanonicalConsolidationPipeline.propose@1"
)
_BASELINE_PROSPECTIVE_REVISION = (
    "prospective:um_mem.memory.prospective.evaluate_prospective_trigger@1"
)


CatalogSnapshot = tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ServicePolicyBundleIdentity:
    """The role-independent canonical key for one local service policy state."""

    repo_root: Path
    workspace_id: str
    backend: str
    profile_id: str = "default"

    def __post_init__(self) -> None:
        root = Path(self.repo_root).expanduser().resolve()
        workspace = normalize_workspace_id(str(self.workspace_id or ""))
        backend = str(self.backend or "").strip().lower()
        profile = str(self.profile_id or "").strip()
        if not workspace:
            raise ValueError("workspace_id is required for a service policy bundle")
        if not _is_identifier(backend):
            raise ValueError("backend must be a safe non-empty identifier")
        if not _is_identifier(profile):
            raise ValueError("profile_id must be a safe non-empty identifier")
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "workspace_id", workspace)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "profile_id", profile)


@dataclass(frozen=True)
class MemoryPolicyBundleRevision:
    """Immutable references to the current concrete UM-MEM policy domains."""

    catalog_revision_id: str
    activation_revision_id: str
    working_context_revision_id: str
    recall_operator_revision_id: str
    consolidation_revision_id: str
    prospective_policy_revision_id: str
    revision_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in _bundle_field_names():
            value = str(getattr(self, name) or "").strip()
            if not _is_identifier(value):
                raise ValueError(f"{name} must be a safe non-empty revision identifier")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "revision_id", _bundle_revision_id(self._revision_mapping()))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "MemoryPolicyBundleRevision":
        fields = _bundle_field_names()
        if set(value) != {*fields, "revision_id"}:
            raise ValueError("policy bundle revision record has unexpected fields")
        bundle = cls(**{name: _required_string(value, name) for name in fields})
        if _required_string(value, "revision_id") != bundle.revision_id:
            raise ValueError("policy bundle revision id does not match its policy references")
        return bundle

    def to_mapping(self) -> dict[str, str]:
        return {**self._revision_mapping(), "revision_id": self.revision_id}

    def _revision_mapping(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in _bundle_field_names()}


@dataclass(frozen=True)
class ActivationReceipt:
    """The content-free, replay-safe result of one requested activation."""

    activation_id_sha256: str
    requested_revision_id: str
    active_revision_id: str
    last_known_good_revision_id: str
    status: str
    reason: str = ""

    def __post_init__(self) -> None:
        if not _SHA256_IDENTIFIER.fullmatch(str(self.activation_id_sha256 or "")):
            raise ValueError("activation_id_sha256 must be a SHA-256 identifier")
        for name in (
            "requested_revision_id",
            "active_revision_id",
            "last_known_good_revision_id",
        ):
            value = str(getattr(self, name) or "").strip()
            if not _SHA256_IDENTIFIER.fullmatch(value):
                raise ValueError(f"{name} must be a SHA-256 revision identifier")
            object.__setattr__(self, name, value)
        status = str(self.status or "").strip()
        if status not in {"activated", "rejected"}:
            raise ValueError("activation receipt status is invalid")
        reason = str(self.reason or "").strip()
        if reason and not _is_identifier(reason):
            raise ValueError("activation receipt reason is invalid")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)

    @property
    def accepted(self) -> bool:
        return self.status == "activated"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ActivationReceipt":
        expected = {
            "activation_id_sha256",
            "requested_revision_id",
            "active_revision_id",
            "last_known_good_revision_id",
            "status",
            "reason",
        }
        if set(value) != expected:
            raise ValueError("activation receipt has unexpected fields")
        return cls(
            **{
                name: _required_string(value, name, allow_empty=name == "reason")
                for name in expected
            }
        )

    def to_mapping(self) -> dict[str, str]:
        return {
            "activation_id_sha256": self.activation_id_sha256,
            "requested_revision_id": self.requested_revision_id,
            "active_revision_id": self.active_revision_id,
            "last_known_good_revision_id": self.last_known_good_revision_id,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _BundlePointer:
    revision_id: str
    bundle: MemoryPolicyBundleRevision
    catalog_snapshot: CatalogSnapshot

    def __post_init__(self) -> None:
        if self.revision_id != self.bundle.revision_id:
            raise ValueError("policy bundle pointer revision does not match its bundle")
        if self.bundle.catalog_revision_id != catalog_revision_id_for_snapshot(
            self.catalog_snapshot
        ):
            raise ValueError("policy bundle pointer catalog snapshot does not match its bundle")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "_BundlePointer":
        if set(value) != {"revision_id", "bundle", "catalog_snapshot"}:
            raise ValueError("policy bundle pointer has unexpected fields")
        bundle_value = value.get("bundle")
        if not isinstance(bundle_value, Mapping):
            raise ValueError("policy bundle pointer bundle is invalid")
        return cls(
            revision_id=_required_string(value, "revision_id"),
            bundle=MemoryPolicyBundleRevision.from_mapping(bundle_value),
            catalog_snapshot=_normalize_catalog_snapshot(value.get("catalog_snapshot")),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "revision_id": self.revision_id,
            "bundle": self.bundle.to_mapping(),
            "catalog_snapshot": _snapshot_to_json(self.catalog_snapshot),
        }


def catalog_snapshot_from_blocks(blocks: Iterable[MemoryBlock]) -> CatalogSnapshot:
    """Return a JSON-safe immutable snapshot from current compiled block definitions."""

    return _normalize_catalog_snapshot(asdict(block) for block in blocks)


def catalog_revision_id_for_snapshot(snapshot: object) -> str:
    normalized = _normalize_catalog_snapshot(snapshot)
    digest = _sha256(_canonical_json(_snapshot_to_json(normalized)))
    return f"catalog:um_mem.memory.blocks.registry.load_memory_blocks@1:sha256:{digest}"


def projection_revision_id_for_snapshot(snapshot: object) -> str:
    """Return the immutable revision of the catalog-derived routing projection."""

    normalized = _normalize_catalog_snapshot(snapshot)
    digest = _sha256(_canonical_json(_snapshot_to_json(normalized)))
    return f"projection:um_mem.memory.blocks.catalog@1:sha256:{digest}"


def block_revision_id_for_snapshot(snapshot: object) -> str:
    """Return the immutable revision of the pinned logical-block definitions."""

    normalized = _normalize_catalog_snapshot(snapshot)
    digest = _sha256(_canonical_json(_snapshot_to_json(normalized)))
    return f"blocks:um_mem.memory.blocks.registry@1:sha256:{digest}"


def baseline_memory_policy_bundle(catalog_snapshot: object) -> MemoryPolicyBundleRevision:
    """Build the explicit baseline that points to the actual current policy domains."""

    return MemoryPolicyBundleRevision(
        catalog_revision_id=catalog_revision_id_for_snapshot(catalog_snapshot),
        activation_revision_id=_BASELINE_ACTIVATION_REVISION,
        working_context_revision_id=_BASELINE_WORKING_CONTEXT_REVISION,
        recall_operator_revision_id=_BASELINE_RECALL_OPERATOR_REVISION,
        consolidation_revision_id=_BASELINE_CONSOLIDATION_REVISION,
        prospective_policy_revision_id=_BASELINE_PROSPECTIVE_REVISION,
    )


class ServicePolicyBundleStore:
    """One local persistent policy state per repository, workspace, profile, and backend."""

    def __init__(
        self,
        identity: ServicePolicyBundleIdentity | str | Path | object,
        *,
        workspace_id: str | None = None,
        profile_id: str | None = None,
        backend: str | None = None,
    ) -> None:
        self.identity = _coerce_identity(
            identity,
            workspace_id=workspace_id,
            profile_id=profile_id,
            backend=backend,
        )
        identity_digest = _sha256(
            _canonical_json(
                {
                    "repo_root_sha256": _sha256(_identity_path(self.identity.repo_root)),
                    "workspace_id_sha256": _sha256(self.identity.workspace_id),
                    "profile_id_sha256": _sha256(self.identity.profile_id),
                    "backend": self.identity.backend,
                }
            )
        )
        self.state_path = (
            self.identity.repo_root
            / ".UM-MEM"
            / "memory"
            / "service_policy_bundles"
            / f"{identity_digest}.json"
        )
        self._lock = threading.RLock()

    def ensure_initialized(
        self,
        bundle: MemoryPolicyBundleRevision,
        catalog_snapshot: object,
    ) -> MemoryPolicyBundleRevision:
        """Create state only when absent; existing state is never auto-replaced."""

        normalized = _normalize_catalog_snapshot(catalog_snapshot)
        _validate_bundle_snapshot(bundle, normalized)
        with self._lock:
            if self.state_path.exists():
                state = self._read_state()
                active, repaired = self._resolve_active_pointer(state)
                if repaired:
                    self._write_state(state)
                return active.bundle
            pointer = _BundlePointer(bundle.revision_id, bundle, normalized)
            state = self._new_state(pointer)
            self._write_state(state)
            return bundle

    def active_bundle(self) -> MemoryPolicyBundleRevision:
        with self._lock:
            state = self._read_state()
            pointer, repaired = self._resolve_active_pointer(state)
            if repaired:
                self._write_state(state)
            return pointer.bundle

    def last_known_good_bundle(self) -> MemoryPolicyBundleRevision:
        with self._lock:
            state = self._read_state()
            pointer, repaired = self._resolve_last_known_good_pointer(state)
            if repaired:
                self._write_state(state)
            return pointer.bundle

    def pin_session(self, session_id: str) -> MemoryPolicyBundleRevision:
        session_hash = _token_hash("session", session_id)
        with self._lock:
            state = self._read_state()
            active, repaired = self._resolve_active_pointer(state)
            pins = _mapping_field(state, "session_pins")
            pinned_id = pins.get(session_hash)
            if isinstance(pinned_id, str):
                pinned = self._pointer_for_revision(state, pinned_id)
                if pinned is None:
                    raise RuntimeError("service policy session pin is invalid")
                if repaired:
                    self._write_state(state)
                return pinned.bundle
            pins[session_hash] = active.revision_id
            self._record_bundle(state, active)
            self._write_state(state)
            return active.bundle

    def pinned_bundle(self, session_id: str) -> MemoryPolicyBundleRevision | None:
        session_hash = _token_hash("session", session_id)
        with self._lock:
            state = self._read_state()
            pins = _mapping_field(state, "session_pins")
            revision_id = pins.get(session_hash)
            if revision_id is None:
                return None
            if not isinstance(revision_id, str):
                raise RuntimeError("service policy session pin is invalid")
            pointer = self._pointer_for_revision(state, revision_id)
            if pointer is None:
                raise RuntimeError("service policy session pin is invalid")
            return pointer.bundle

    def stage_bundle(
        self,
        bundle: MemoryPolicyBundleRevision,
        catalog_snapshot: object,
    ) -> MemoryPolicyBundleRevision:
        """Persist a validated candidate without changing the active pointer."""

        if not isinstance(bundle, MemoryPolicyBundleRevision):
            raise TypeError("bundle must be a MemoryPolicyBundleRevision")
        snapshot = _normalize_catalog_snapshot(catalog_snapshot)
        _validate_bundle_snapshot(bundle, snapshot)
        pointer = _BundlePointer(bundle.revision_id, bundle, snapshot)
        with self._lock:
            state = self._read_state()
            active, _ = self._resolve_active_pointer(state)
            self._record_bundle(state, active)
            self._record_bundle(state, pointer)
            self._write_state(state)
        return bundle

    def pin_session_to_bundle(
        self,
        session_id: str,
        bundle_revision_id: str | MemoryPolicyBundleRevision,
    ) -> MemoryPolicyBundleRevision:
        """Atomically pin one whole session to a staged or active bundle."""

        session_hash = _token_hash("session", session_id)
        revision_id = _revision_id_value(bundle_revision_id)
        with self._lock:
            state = self._read_state()
            pointer = self._pointer_for_revision(state, revision_id)
            if pointer is None:
                raise ValueError("policy bundle revision is not staged")
            _mapping_field(state, "session_pins")[session_hash] = revision_id
            self._record_bundle(state, pointer)
            self._write_state(state)
            return pointer.bundle

    def reassign_bundle_pins(
        self,
        *,
        from_revision_id: str | MemoryPolicyBundleRevision,
        to_revision_id: str | MemoryPolicyBundleRevision,
    ) -> int:
        """Move hashed session pins between persisted bundles without raw identities."""

        source_id = _revision_id_value(from_revision_id)
        target_id = _revision_id_value(to_revision_id)
        with self._lock:
            state = self._read_state()
            target = self._pointer_for_revision(state, target_id)
            if target is None:
                raise ValueError("target policy bundle revision is not persisted")
            if self._pointer_for_revision(state, source_id) is None:
                raise ValueError("source policy bundle revision is not persisted")
            pins = _mapping_field(state, "session_pins")
            reassigned = 0
            for session_hash, revision_id in tuple(pins.items()):
                if revision_id == source_id:
                    pins[session_hash] = target_id
                    reassigned += 1
            self._record_bundle(state, target)
            self._write_state(state)
            return reassigned

    def snapshot_for(self, bundle_revision_id: str | MemoryPolicyBundleRevision) -> CatalogSnapshot:
        revision_id = _revision_id_value(bundle_revision_id)
        with self._lock:
            state = self._read_state()
            pointer = self._pointer_for_revision(state, revision_id)
            if pointer is None:
                raise ValueError("policy bundle revision has no valid persisted catalog snapshot")
            return _copy_snapshot(pointer.catalog_snapshot)

    def bundle_for(
        self,
        bundle_revision_id: str | MemoryPolicyBundleRevision,
    ) -> MemoryPolicyBundleRevision:
        """Return one validated persisted bundle without changing any pointer or pin."""

        revision_id = _revision_id_value(bundle_revision_id)
        with self._lock:
            state = self._read_state()
            pointer = self._pointer_for_revision(state, revision_id)
            if pointer is None:
                raise ValueError("policy bundle revision is not persisted")
            return pointer.bundle

    def activate(
        self,
        bundle: MemoryPolicyBundleRevision,
        *,
        activation_id: str,
        catalog_snapshot: object | None = None,
    ) -> ActivationReceipt:
        if not isinstance(bundle, MemoryPolicyBundleRevision):
            raise TypeError("bundle must be a MemoryPolicyBundleRevision")
        activation_hash = f"sha256:{_token_hash('activation', activation_id)}"
        with self._lock:
            state = self._read_state()
            active, repaired = self._resolve_active_pointer(state)
            receipts = _mapping_field(state, "activation_receipts")
            existing = receipts.get(activation_hash)
            if existing is not None:
                if not isinstance(existing, Mapping):
                    raise RuntimeError("service policy activation receipt is invalid")
                receipt = ActivationReceipt.from_mapping(existing)
                if receipt.requested_revision_id != bundle.revision_id:
                    raise ValueError("service policy activation receipt conflict")
                if repaired:
                    self._write_state(state)
                return receipt

            try:
                snapshot = self._activation_snapshot(state, bundle, catalog_snapshot)
                _validate_bundle_snapshot(bundle, snapshot)
            except ValueError as exc:
                receipt = ActivationReceipt(
                    activation_id_sha256=activation_hash,
                    requested_revision_id=bundle.revision_id,
                    active_revision_id=active.revision_id,
                    last_known_good_revision_id=active.revision_id,
                    status="rejected",
                    reason=_activation_rejection_reason(exc),
                )
                receipts[activation_hash] = receipt.to_mapping()
                if repaired:
                    self._record_bundle(state, active)
                self._write_state(state)
                return receipt

            pointer = _BundlePointer(bundle.revision_id, bundle, snapshot)
            self._record_bundle(state, pointer)
            pointers = _mapping_field(state, "pointers")
            pointers["active"] = pointer.to_mapping()
            pointers["last_known_good"] = pointer.to_mapping()
            receipt = ActivationReceipt(
                activation_id_sha256=activation_hash,
                requested_revision_id=bundle.revision_id,
                active_revision_id=bundle.revision_id,
                last_known_good_revision_id=bundle.revision_id,
                status="activated",
            )
            receipts[activation_hash] = receipt.to_mapping()
            self._write_state(state)
            return receipt

    def _activation_snapshot(
        self,
        state: dict[str, object],
        bundle: MemoryPolicyBundleRevision,
        catalog_snapshot: object | None,
    ) -> CatalogSnapshot:
        if catalog_snapshot is not None:
            return _normalize_catalog_snapshot(catalog_snapshot)
        existing = self._pointer_for_revision(state, bundle.revision_id)
        if existing is not None:
            return existing.catalog_snapshot
        active, _ = self._resolve_active_pointer(state)
        if active.bundle.catalog_revision_id == bundle.catalog_revision_id:
            return active.catalog_snapshot
        raise ValueError("catalog_snapshot_required_for_changed_catalog")

    def _new_state(self, pointer: _BundlePointer) -> dict[str, object]:
        state: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "identity": self._identity_record(),
            "bundles": {},
            "catalog_snapshots": {},
            "pointers": {
                "active": pointer.to_mapping(),
                "last_known_good": pointer.to_mapping(),
            },
            "session_pins": {},
            "activation_receipts": {},
        }
        self._record_bundle(state, pointer)
        return state

    def _record_bundle(self, state: dict[str, object], pointer: _BundlePointer) -> None:
        bundles = _mapping_field(state, "bundles")
        snapshots = _mapping_field(state, "catalog_snapshots")
        bundles[pointer.revision_id] = pointer.bundle.to_mapping()
        snapshots[pointer.revision_id] = _snapshot_to_json(pointer.catalog_snapshot)

    def _resolve_active_pointer(self, state: dict[str, object]) -> tuple[_BundlePointer, bool]:
        pointer = self._named_pointer(state, "active")
        if pointer is not None:
            self._record_bundle(state, pointer)
            return pointer, False
        fallback = self._named_pointer(state, "last_known_good")
        if fallback is None:
            raise RuntimeError("service policy state has no valid active or last-known-good bundle")
        _mapping_field(state, "pointers")["active"] = fallback.to_mapping()
        self._record_bundle(state, fallback)
        return fallback, True

    def _resolve_last_known_good_pointer(
        self, state: dict[str, object]
    ) -> tuple[_BundlePointer, bool]:
        pointer = self._named_pointer(state, "last_known_good")
        if pointer is not None:
            self._record_bundle(state, pointer)
            return pointer, False
        active = self._named_pointer(state, "active")
        if active is None:
            raise RuntimeError("service policy state has no valid active or last-known-good bundle")
        _mapping_field(state, "pointers")["last_known_good"] = active.to_mapping()
        self._record_bundle(state, active)
        return active, True

    def _named_pointer(self, state: dict[str, object], name: str) -> _BundlePointer | None:
        pointers = _mapping_field(state, "pointers")
        raw = pointers.get(name)
        if not isinstance(raw, Mapping):
            return None
        try:
            return _BundlePointer.from_mapping(raw)
        except ValueError:
            return None

    def _pointer_for_revision(
        self, state: dict[str, object], revision_id: str
    ) -> _BundlePointer | None:
        bundles = _mapping_field(state, "bundles")
        snapshots = _mapping_field(state, "catalog_snapshots")
        raw_bundle = bundles.get(revision_id)
        raw_snapshot = snapshots.get(revision_id)
        if isinstance(raw_bundle, Mapping):
            try:
                return _BundlePointer(
                    revision_id=revision_id,
                    bundle=MemoryPolicyBundleRevision.from_mapping(raw_bundle),
                    catalog_snapshot=_normalize_catalog_snapshot(raw_snapshot),
                )
            except ValueError:
                pass
        for name in ("active", "last_known_good"):
            pointer = self._named_pointer(state, name)
            if pointer is not None and pointer.revision_id == revision_id:
                self._record_bundle(state, pointer)
                return pointer
        return None

    def _identity_record(self) -> dict[str, str]:
        return {
            "repo_root_sha256": _sha256(_identity_path(self.identity.repo_root)),
            "workspace_id_sha256": _sha256(self.identity.workspace_id),
            "profile_id_sha256": _sha256(self.identity.profile_id),
            "backend": self.identity.backend,
        }

    def _read_state(self) -> dict[str, object]:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError("service policy state has not been initialized") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("service policy state is unreadable") from exc
        if not isinstance(raw, dict):
            raise RuntimeError("service policy state is invalid")
        state = dict(raw)
        if state.get("schema_version") != _SCHEMA_VERSION:
            raise RuntimeError("service policy state schema is unsupported")
        identity = state.get("identity")
        if not isinstance(identity, Mapping) or dict(identity) != self._identity_record():
            raise RuntimeError("service policy state identity is not authorized")
        for field_name in (
            "bundles",
            "catalog_snapshots",
            "pointers",
            "session_pins",
            "activation_receipts",
        ):
            _mapping_field(state, field_name)
        return state

    def _write_state(self, state: Mapping[str, object]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.state_path, state)


def _coerce_identity(
    identity: ServicePolicyBundleIdentity | str | Path | object,
    *,
    workspace_id: str | None,
    profile_id: str | None,
    backend: str | None,
) -> ServicePolicyBundleIdentity:
    if isinstance(identity, ServicePolicyBundleIdentity):
        if workspace_id is not None or profile_id is not None or backend is not None:
            raise ValueError(
                "workspace_id, profile_id, and backend belong in the supplied service identity"
            )
        return identity
    if workspace_id is None:
        candidate_workspace = getattr(identity, "workspace_id", None)
    else:
        candidate_workspace = workspace_id
    if backend is None:
        candidate_backend = getattr(identity, "backend", None)
    else:
        candidate_backend = backend
    if profile_id is None:
        candidate_profile = getattr(identity, "profile_id", "default")
    else:
        candidate_profile = profile_id
    candidate_root = getattr(identity, "repo_root", identity)
    if not isinstance(candidate_root, (str, Path)):
        raise ValueError("repo_root must be a path string")
    if (
        not isinstance(candidate_workspace, str)
        or not isinstance(candidate_profile, str)
        or not isinstance(candidate_backend, str)
    ):
        raise ValueError(
            "workspace_id, profile_id, and backend are required for a service policy bundle"
        )
    return ServicePolicyBundleIdentity(
        repo_root=Path(candidate_root),
        workspace_id=candidate_workspace,
        profile_id=candidate_profile,
        backend=candidate_backend,
    )


def _bundle_field_names() -> tuple[str, ...]:
    return (
        "catalog_revision_id",
        "activation_revision_id",
        "working_context_revision_id",
        "recall_operator_revision_id",
        "consolidation_revision_id",
        "prospective_policy_revision_id",
    )


def _bundle_revision_id(values: Mapping[str, str]) -> str:
    return f"sha256:{_sha256(_canonical_json(dict(values)))}"


def _validate_bundle_snapshot(
    bundle: MemoryPolicyBundleRevision, snapshot: CatalogSnapshot
) -> None:
    if not isinstance(bundle, MemoryPolicyBundleRevision):
        raise TypeError("bundle must be a MemoryPolicyBundleRevision")
    if bundle.catalog_revision_id != catalog_revision_id_for_snapshot(snapshot):
        raise ValueError("catalog_snapshot_revision_mismatch")


def _normalize_catalog_snapshot(value: object) -> CatalogSnapshot:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise ValueError("catalog_snapshot must be a non-empty iterable of block mappings")
    rows: list[dict[str, object]] = []
    block_ids: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("catalog_snapshot entries must be mappings")
        row = _json_safe_mapping(item)
        block_id = row.get("block_id")
        if not isinstance(block_id, str) or not _is_identifier(block_id):
            raise ValueError("catalog_snapshot entries require safe block_id values")
        if block_id in block_ids:
            raise ValueError("catalog_snapshot block_id values must be unique")
        block_ids.add(block_id)
        rows.append(row)
    if not rows:
        raise ValueError("catalog_snapshot must not be empty")
    rows.sort(key=lambda item: str(item["block_id"]))
    return tuple(rows)


def _json_safe_mapping(value: Mapping[object, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not _is_identifier(raw_key):
            raise ValueError("catalog_snapshot keys must be safe identifiers")
        if raw_key.casefold() in _CONTENT_KEYS:
            raise ValueError("catalog_snapshot must remain content-free")
        result[raw_key] = _json_safe_value(raw_value)
    return {key: result[key] for key in sorted(result)}


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("catalog_snapshot cannot contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    raise ValueError("catalog_snapshot must contain JSON-safe values")


def _snapshot_to_json(snapshot: CatalogSnapshot) -> list[dict[str, object]]:
    return json.loads(_canonical_json(list(snapshot)))


def _copy_snapshot(snapshot: CatalogSnapshot) -> CatalogSnapshot:
    return _normalize_catalog_snapshot(_snapshot_to_json(snapshot))


def _mapping_field(state: dict[str, object], field_name: str) -> dict[str, object]:
    value = state.get(field_name)
    if not isinstance(value, dict):
        raise RuntimeError(f"service policy state {field_name} is invalid")
    return value


def _required_string(
    value: Mapping[str, object],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    raw = value.get(field_name)
    if not isinstance(raw, str):
        raise ValueError(f"{field_name} must be a string")
    result = raw.strip()
    if not result and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _revision_id_value(value: str | MemoryPolicyBundleRevision) -> str:
    revision_id = (
        value.revision_id if isinstance(value, MemoryPolicyBundleRevision) else str(value or "")
    )
    if not _SHA256_IDENTIFIER.fullmatch(revision_id):
        raise ValueError("bundle_revision_id must be a SHA-256 revision identifier")
    return revision_id


def _activation_rejection_reason(exc: ValueError) -> str:
    reason = str(exc or "").strip()
    return reason if _is_identifier(reason) else "invalid_policy_bundle_activation"


def _token_hash(kind: str, value: str) -> str:
    raw = str(value or "")
    if not raw or len(raw) > 4096:
        raise ValueError(f"{kind}_id is required and must be at most 4096 characters")
    return _sha256(f"{kind}:{raw}")


def _identity_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _is_identifier(value: str) -> bool:
    return bool(_IDENTIFIER.fullmatch(str(value or "")))


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, path)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
