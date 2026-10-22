import hashlib
import json
from dataclasses import replace

import pytest

from um_mem.memory.service_policy import (
    ServicePolicyBundleStore,
    baseline_memory_policy_bundle,
)


def _catalog_snapshot() -> tuple[dict[str, object], ...]:
    return (
        {
            "block_id": "memory_root_index",
            "block_type": "memory_index",
            "temperature": "warm",
            "scope": "workspace",
            "authority": "advisory",
            "delivery_policy": "always_on",
            "retention": "durable",
            "tags": ["memory", "index"],
            "child_ids": ["working_memory_index"],
            "index_only": True,
            "token_budget": 0,
        },
        {
            "block_id": "working_memory_index",
            "block_type": "working_set_index",
            "temperature": "hot",
            "scope": "session",
            "authority": "advisory",
            "delivery_policy": "working_set",
            "retention": "session",
            "tags": ["run", "working_memory"],
            "parent_id": "memory_root_index",
            "index_only": True,
            "token_budget": 0,
        },
    )


def _store(tmp_path) -> tuple[ServicePolicyBundleStore, object, tuple[dict[str, object], ...]]:
    snapshot = _catalog_snapshot()
    bundle = baseline_memory_policy_bundle(snapshot)
    store = ServicePolicyBundleStore(tmp_path, workspace_id="release", backend="mempalace")
    assert store.ensure_initialized(bundle, snapshot) == bundle
    return store, bundle, snapshot


def test_initialize_pins_explicit_baseline_and_hides_raw_session(tmp_path) -> None:
    store, bundle, snapshot = _store(tmp_path)

    pinned = store.pin_session("session containing user text")

    assert pinned == bundle
    assert pinned.catalog_revision_id.startswith(
        "catalog:um_mem.memory.blocks.registry.load_memory_blocks@1:sha256:"
    )
    assert pinned.activation_revision_id == (
        "activation:um_mem.memory.selection.MemorySelectionPolicy.eligible_for_overlay@1"
    )
    assert pinned.working_context_revision_id == (
        "working_context:um_mem.memory.working_continuity.compile_continuation_packet@1"
    )
    assert pinned.recall_operator_revision_id == (
        "recall_operator:um_mem.memory.recall_plan.RecallPlan.compile@2"
    )
    assert pinned.consolidation_revision_id == (
        "consolidation:um_mem.memory.consolidation.CanonicalConsolidationPipeline.propose@1"
    )
    assert pinned.prospective_policy_revision_id == (
        "prospective:um_mem.memory.prospective.evaluate_prospective_trigger@1"
    )
    assert store.snapshot_for(pinned.revision_id) == snapshot

    stored = store.state_path.read_text(encoding="utf-8")
    assert "session containing user text" not in stored
    assert hashlib.sha256(b"session:session containing user text").hexdigest() in stored


def test_activation_receipt_is_idempotent_for_caller_activation_id(tmp_path) -> None:
    store, bundle, snapshot = _store(tmp_path)
    candidate = replace(
        bundle,
        consolidation_revision_id="consolidation:um_mem.memory.consolidation.CanonicalConsolidationPipeline.propose@2",
    )

    first = store.activate(
        candidate, activation_id="activate-consolidation", catalog_snapshot=snapshot
    )
    second = store.activate(
        candidate, activation_id="activate-consolidation", catalog_snapshot=snapshot
    )
    restarted = ServicePolicyBundleStore(store.identity)
    replayed = restarted.activate(
        candidate, activation_id="activate-consolidation", catalog_snapshot=snapshot
    )

    assert first == second == replayed
    assert first.accepted is True
    assert restarted.active_bundle() == candidate
    assert restarted.last_known_good_bundle() == candidate
    with pytest.raises(ValueError, match="conflict"):
        restarted.activate(
            bundle, activation_id="activate-consolidation", catalog_snapshot=snapshot
        )


def test_rejected_catalog_activation_keeps_last_known_good_pointer(tmp_path) -> None:
    store, baseline, snapshot = _store(tmp_path)
    candidate = replace(
        baseline,
        catalog_revision_id=(
            "catalog:um_mem.memory.blocks.registry.load_memory_blocks@1:sha256:" + "0" * 64
        ),
    )

    receipt = store.activate(
        candidate,
        activation_id="reject-invalid-catalog",
        catalog_snapshot=snapshot,
    )

    assert receipt.accepted is False
    assert receipt.status == "rejected"
    assert receipt.reason == "catalog_snapshot_revision_mismatch"
    assert store.active_bundle() == baseline
    assert store.last_known_good_bundle() == baseline


def test_reload_preserves_session_pin_and_recovers_active_from_lkg(tmp_path) -> None:
    store, baseline, snapshot = _store(tmp_path)
    assert store.pin_session("stable-session") == baseline
    candidate = replace(
        baseline,
        prospective_policy_revision_id="prospective:um_mem.memory.prospective.evaluate_prospective_trigger@2",
    )
    assert store.activate(
        candidate, activation_id="activate-prospective", catalog_snapshot=snapshot
    ).accepted

    state = json.loads(store.state_path.read_text(encoding="utf-8"))
    state["pointers"]["active"]["revision_id"] = "sha256:" + "0" * 64
    store.state_path.write_text(json.dumps(state), encoding="utf-8")

    reloaded = ServicePolicyBundleStore(tmp_path, workspace_id="release", backend="mempalace")
    assert reloaded.pinned_bundle("stable-session") == baseline
    assert reloaded.active_bundle() == candidate
    assert reloaded.snapshot_for(candidate.revision_id) == snapshot

    repaired = json.loads(reloaded.state_path.read_text(encoding="utf-8"))
    assert repaired["pointers"]["active"]["revision_id"] == candidate.revision_id
    assert repaired["pointers"]["last_known_good"]["revision_id"] == candidate.revision_id
