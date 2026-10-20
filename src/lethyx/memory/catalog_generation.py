"""Atomic catalog generations with session canaries and lossless rollback."""

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from lethyx.memory.service_policy import (
    ActivationReceipt,
    MemoryPolicyBundleRevision,
    ServicePolicyBundleStore,
)


GenerationStatus = Literal["staged", "canary", "active", "rolled_back"]


def canonical_record_set_sha256(record_ids: Iterable[str]) -> str:
    """Fingerprint canonical identities without reading or persisting their content."""

    normalized = sorted({_required(item, field_name="canonical record id") for item in record_ids})
    payload = json.dumps(normalized, ensure_ascii=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class CatalogGeneration:
    generation_id: str
    bundle_revision_id: str
    prior_bundle_revision_id: str
    canonical_record_set_sha256: str
    status: GenerationStatus
    canary_session_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("generation_id", "bundle_revision_id", "prior_bundle_revision_id"):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), field_name=name.replace("_", " ")),
            )
        object.__setattr__(
            self,
            "canonical_record_set_sha256",
            _sha256_identifier(
                self.canonical_record_set_sha256,
                field_name="canonical record set sha256",
            ),
        )
        object.__setattr__(self, "status", _generation_status(self.status))
        sessions = tuple(
            sorted(
                {
                    _sha256_identifier(item, field_name="canary session sha256")
                    for item in self.canary_session_sha256s
                }
            )
        )
        object.__setattr__(self, "canary_session_sha256s", sessions)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CatalogGeneration":
        expected = {
            "generation_id",
            "bundle_revision_id",
            "prior_bundle_revision_id",
            "canonical_record_set_sha256",
            "status",
            "canary_session_sha256s",
        }
        if set(value) != expected:
            raise ValueError("catalog generation record has unexpected fields")
        sessions = value.get("canary_session_sha256s")
        if not isinstance(sessions, list) or not all(isinstance(item, str) for item in sessions):
            raise ValueError("catalog generation canary sessions are invalid")
        return cls(
            generation_id=_required(value.get("generation_id"), field_name="generation id"),
            bundle_revision_id=_required(
                value.get("bundle_revision_id"), field_name="bundle revision id"
            ),
            prior_bundle_revision_id=_required(
                value.get("prior_bundle_revision_id"), field_name="prior bundle revision id"
            ),
            canonical_record_set_sha256=_sha256_identifier(
                value.get("canonical_record_set_sha256"),
                field_name="canonical record set sha256",
            ),
            status=_generation_status(value.get("status")),
            canary_session_sha256s=tuple(sorted(set(sessions))),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "bundle_revision_id": self.bundle_revision_id,
            "prior_bundle_revision_id": self.prior_bundle_revision_id,
            "canonical_record_set_sha256": self.canonical_record_set_sha256,
            "status": self.status,
            "canary_session_sha256s": list(self.canary_session_sha256s),
        }


class CatalogGenerationController:
    """Compile, canary, activate, and roll back immutable catalog generations."""

    def __init__(
        self,
        policy_store: ServicePolicyBundleStore,
        *,
        canonical_record_ids: Callable[[], Iterable[str]],
    ) -> None:
        if not isinstance(policy_store, ServicePolicyBundleStore):
            raise TypeError("policy_store must be a ServicePolicyBundleStore")
        if not callable(canonical_record_ids):
            raise TypeError("canonical_record_ids must be callable")
        self.policy_store = policy_store
        self._canonical_record_ids = canonical_record_ids
        self.path = policy_store.state_path.with_name(
            f"{policy_store.state_path.stem}.catalog-generations.json"
        )
        self._lock = threading.RLock()

    def compile(
        self,
        bundle: MemoryPolicyBundleRevision,
        catalog_snapshot: object,
    ) -> CatalogGeneration:
        """Validate and atomically stage a complete generation without activation."""

        prior = self.policy_store.active_bundle()
        before = self._canonical_fingerprint()
        self.policy_store.stage_bundle(bundle, catalog_snapshot)
        self._assert_canonical_preserved(before)
        generation_id = _generation_id(bundle.revision_id, prior.revision_id)
        generation = CatalogGeneration(
            generation_id=generation_id,
            bundle_revision_id=bundle.revision_id,
            prior_bundle_revision_id=prior.revision_id,
            canonical_record_set_sha256=before,
            status="staged",
        )
        with self._lock:
            state = self._read_state()
            existing = state.get(generation_id)
            if existing is not None:
                current = CatalogGeneration.from_mapping(_mapping(existing))
                if (
                    current.bundle_revision_id != bundle.revision_id
                    or current.prior_bundle_revision_id != prior.revision_id
                ):
                    raise ValueError("catalog generation identity conflict")
                return current
            state[generation_id] = generation.to_mapping()
            self._write_state(state)
        return generation

    def start_canary(self, generation_id: str, *, session_id: str) -> CatalogGeneration:
        """Pin exactly one whole session to a staged candidate generation."""

        with self._lock:
            state = self._read_state()
            generation = self._generation_from_state(state, generation_id)
            if generation.status not in {"staged", "canary"}:
                raise ValueError("catalog generation is not available for canary")
            before = self._canonical_fingerprint()
            self.policy_store.pin_session_to_bundle(session_id, generation.bundle_revision_id)
            self._assert_canonical_preserved(before)
            session_hash = _token_sha256("session", session_id)
            updated = replace(
                generation,
                status="canary",
                canary_session_sha256s=tuple(
                    sorted({*generation.canary_session_sha256s, session_hash})
                ),
            )
            state[generation.generation_id] = updated.to_mapping()
            self._write_state(state)
            return updated

    def rollback_canary(self, generation_id: str, *, session_id: str) -> CatalogGeneration:
        """Move a canary session back to its exact pre-generation bundle."""

        with self._lock:
            state = self._read_state()
            generation = self._generation_from_state(state, generation_id)
            session_hash = _token_sha256("session", session_id)
            if session_hash not in generation.canary_session_sha256s:
                raise ValueError("session is not pinned to this catalog generation")
            before = self._canonical_fingerprint()
            self.policy_store.pin_session_to_bundle(
                session_id,
                generation.prior_bundle_revision_id,
            )
            self._assert_canonical_preserved(before)
            remaining = tuple(
                item for item in generation.canary_session_sha256s if item != session_hash
            )
            updated = replace(
                generation,
                status="canary" if remaining else "staged",
                canary_session_sha256s=remaining,
            )
            state[generation.generation_id] = updated.to_mapping()
            self._write_state(state)
            return updated

    def activate(self, generation_id: str, *, activation_id: str) -> ActivationReceipt:
        """Atomically swap the active catalog pointer to a compiled generation."""

        with self._lock:
            state = self._read_state()
            generation = self._generation_from_state(state, generation_id)
            if generation.status not in {"staged", "canary", "active"}:
                raise ValueError("catalog generation cannot be activated")
            bundle = self._bundle(generation.bundle_revision_id)
            snapshot = self.policy_store.snapshot_for(bundle)
            before = self._canonical_fingerprint()
            receipt = self.policy_store.activate(
                bundle,
                activation_id=activation_id,
                catalog_snapshot=snapshot,
            )
            self._assert_canonical_preserved(before)
            if receipt.accepted:
                updated = replace(generation, status="active")
                state[generation.generation_id] = updated.to_mapping()
                self._write_state(state)
            return receipt

    def rollback(self, generation_id: str, *, activation_id: str) -> ActivationReceipt:
        """Atomically restore the exact prior catalog without touching records."""

        with self._lock:
            state = self._read_state()
            generation = self._generation_from_state(state, generation_id)
            active_id = self.policy_store.active_bundle().revision_id
            if active_id not in {
                generation.bundle_revision_id,
                generation.prior_bundle_revision_id,
            }:
                raise ValueError("active catalog no longer belongs to this generation")
            prior = self._bundle(generation.prior_bundle_revision_id)
            snapshot = self.policy_store.snapshot_for(prior)
            before = self._canonical_fingerprint()
            receipt = self.policy_store.activate(
                prior,
                activation_id=activation_id,
                catalog_snapshot=snapshot,
            )
            if receipt.accepted:
                self.policy_store.reassign_bundle_pins(
                    from_revision_id=generation.bundle_revision_id,
                    to_revision_id=generation.prior_bundle_revision_id,
                )
            self._assert_canonical_preserved(before)
            if receipt.accepted:
                updated = replace(
                    generation,
                    status="rolled_back",
                    canary_session_sha256s=(),
                )
                state[generation.generation_id] = updated.to_mapping()
                self._write_state(state)
            return receipt

    def generation(self, generation_id: str) -> CatalogGeneration:
        with self._lock:
            return self._generation_from_state(self._read_state(), generation_id)

    def _bundle(self, revision_id: str) -> MemoryPolicyBundleRevision:
        return self.policy_store.bundle_for(revision_id)

    def _canonical_fingerprint(self) -> str:
        return canonical_record_set_sha256(self._canonical_record_ids())

    def _assert_canonical_preserved(self, before: str) -> None:
        if self._canonical_fingerprint() != before:
            raise RuntimeError("catalog generation operation changed canonical records")

    def _generation_from_state(
        self,
        state: Mapping[str, object],
        generation_id: str,
    ) -> CatalogGeneration:
        value = state.get(_required(generation_id, field_name="generation id"))
        if value is None:
            raise KeyError("unknown catalog generation")
        return CatalogGeneration.from_mapping(_mapping(value))

    def _read_state(self) -> dict[str, object]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("catalog generation state is invalid") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise RuntimeError("catalog generation state is invalid")
        generations = value.get("generations")
        if not isinstance(generations, dict):
            raise RuntimeError("catalog generation state is invalid")
        return dict(generations)

    def _write_state(self, generations: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"schema_version": 1, "generations": dict(generations)},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _generation_id(bundle_revision_id: str, prior_revision_id: str) -> str:
    return _token_sha256("catalog-generation", f"{prior_revision_id}|{bundle_revision_id}")


def _token_sha256(domain: str, value: object) -> str:
    text = _required(value, field_name=domain)
    digest = hashlib.sha256(f"{domain}:{text}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _required(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 512:
        raise ValueError(f"{field_name} must be a bounded non-empty identifier")
    return text


def _sha256_identifier(value: object, *, field_name: str) -> str:
    text = _required(value, field_name=field_name)
    if len(text) != 71 or not text.startswith("sha256:"):
        raise ValueError(f"{field_name} must be a SHA-256 identifier")
    try:
        int(text[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a SHA-256 identifier") from exc
    return text


def _generation_status(value: object) -> GenerationStatus:
    status = str(value or "").strip()
    if status not in {"staged", "canary", "active", "rolled_back"}:
        raise ValueError("catalog generation status is invalid")
    return cast(GenerationStatus, status)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("catalog generation record is invalid")
    return value
