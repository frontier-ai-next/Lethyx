import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from lethyx import MemoryClient
from lethyx.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from lethyx.memory.backends.contracts import block_record_to_memory_hit, block_to_json
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.blocks.catalog import DynamicLabel, RefinementCase, compile_semantic_catalog
from lethyx.memory.blocks.registry import load_memory_blocks
from lethyx.memory.service_policy import (
    ServicePolicyBundleStore,
    baseline_memory_policy_bundle,
    catalog_snapshot_from_blocks,
)


def _evidence(name: str) -> EvidenceRef:
    return EvidenceRef(
        ref_type="test_run",
        ref_id=name,
        produced_by="verifier",
        producer_id=f"verifier:phase-4-exit:{name}",
    )


def test_real_project_corpus_indexes_edges_exact_ids_and_rebuild_are_deterministic(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    corpus = (
        (
            "README.md",
            "project_fact",
            "stable_project_facts",
            "semantic",
            "README contract",
        ),
        (
            "pyproject.toml",
            "configuration_state",
            "mutable_project_state",
            "state",
            "Packaging state",
        ),
        (
            "docs/lethyx/operator_guide.md",
            "episode",
            "workspace_episodes",
            "episodic",
            "Operator episode",
        ),
        (
            "docs/lethyx/quickstart.md",
            "reflection_candidate",
            "reflection_candidates",
            "reflection",
            "Quickstart lesson candidate",
        ),
        (
            "src/lethyx/api.py",
            "code_chunk",
            "code_resources",
            "resource",
            "Client resource",
        ),
        (
            "docs/lethyx/memory_architecture.md",
            "reflection_candidate",
            "reflection_candidates",
            "reflection",
            "Architecture reflection",
        ),
    )
    backend = BackendSQLiteIndex(
        tmp_path / "index.sqlite",
        backend_id="test",
        workspace_id="release",
    )
    client = MemoryClient(tmp_path, workspace_id="release", backend_instance=backend)
    expected_memberships: dict[str, str] = {}
    saved_ids: list[str] = []
    try:
        for ordinal, (
            relative_path,
            kind,
            requested_block_id,
            membership,
            title,
        ) in enumerate(corpus):
            source = repository / relative_path
            result = client.remember(
                source.read_text(encoding="utf-8")[:800],
                title=title,
                kind=kind,
                requested_block_id=requested_block_id,
                evidence_refs=[_evidence(f"corpus-{ordinal}")],
                metadata={
                    "path": relative_path,
                    "symbol": "MemoryClient" if relative_path.endswith("api.py") else title,
                    "repository_id": "Lethyx",
                },
                repository_id="Lethyx",
            )
            assert result.saved is True
            record = backend.get_block(result.memory_id, workspace_id="release")
            assert record is not None
            assert record.block_membership is not None
            logical_block_id = record.block_membership.primary_block_id
            expected_memberships[logical_block_id] = membership
            saved_ids.append(record.block_id)

        catalog = compile_semantic_catalog(load_memory_blocks(tmp_path, "release"))
        observed_memberships: set[str] = set()
        for logical_block_id, membership in expected_memberships.items():
            descriptor_memberships = catalog.descriptor(logical_block_id).semantic_memberships
            assert membership in descriptor_memberships
            observed_memberships.update(descriptor_memberships)
        observed_memberships.update(catalog.descriptor("verified_lessons").semantic_memberships)
        assert {"semantic", "state", "episodic", "procedural", "resource", "reflection"} <= (
            observed_memberships
        )

        node_types = {node.node_type for node in backend.export_index_nodes()}
        assert {"block_descriptor", "project_structure"} <= node_types
        assert backend.export_edges() == []
        assert all(
            backend.get_block(block_id, workspace_id="release").evidence_refs
            for block_id in saved_ids
        )
        assert backend.exact_index_terms(saved_ids[4])["path"] == ("src/lethyx/api.py",)

        exact = backend.recall(
            {
                "workspace_id": "release",
                "paths": ["src/lethyx/api.py"],
                "symbols": ["MemoryClient"],
                "include_candidates": True,
                "source_horizon": "durable",
                "scope_chain": ["workspace:release"],
                "limit": 20,
            }
        )
        assert len(exact) == 1
        assert exact[0].metadata["path"] == "src/lethyx/api.py"
        assert (
            client.recall(
                "unrelated",
                record_ids=[saved_ids[0]],
                search_all=True,
                include_candidates=True,
            )[0].metadata["block_id"]
            == saved_ids[0]
        )
        assert (
            client.recall(
                "README",
                record_ids=["missing-explicit-record"],
                search_all=True,
                include_candidates=True,
            )
            == []
        )

        canonical_before = json.dumps(
            [block_to_json(record) for record in backend.export_blocks()],
            sort_keys=True,
        )
        first_rebuild = client.rebuild_indexes()
        second_rebuild = client.rebuild_indexes()
        canonical_after = json.dumps(
            [block_to_json(record) for record in backend.export_blocks()],
            sort_keys=True,
        )

        assert first_rebuild["record_count"] == len(corpus)
        assert second_rebuild["index_sha256"] == first_rebuild["index_sha256"]
        assert canonical_after == canonical_before
    finally:
        backend.close()


class _ScopedBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.record = MemoryBlockRecord(
            block_id="scoped-record",
            title="Scoped release evidence",
            content="The release marker is cobalt.",
            kind="project_fact",
            status="candidate",
            scope="workspace",
            authority="advisory",
            workspace_id="alpha",
            block_membership=RecordBlockMembership(
                primary_block_id="stable_project_facts",
                catalog_revision_id="catalog:test",
            ),
        )

    def select_index_nodes(self, **kwargs: Any) -> list[Any]:
        del kwargs
        return []

    def recall_evidence(self, query: dict[str, Any]) -> list[Any]:
        self.calls.append(dict(query))
        if query.get("logical_block_ids") == ["stable_project_facts"]:
            return [block_record_to_memory_hit(self.record)]
        return []


def test_explicit_block_scope_reaches_backend_once_without_widening(tmp_path: Path) -> None:
    backend = _ScopedBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    plan = client.compile_recall_plan(
        "project release architecture",
        logical_block_ids=["stable_project_facts"],
        include_candidates=True,
    )

    assert plan is not None
    assert plan.logical_block_ids == ("stable_project_facts",)
    hits = client.recall(plan.query, recall_plan=plan, include_candidates=True)

    assert [hit.metadata["block_id"] for hit in hits] == ["scoped-record"]
    assert len(backend.calls) == 1
    assert backend.calls[0]["logical_block_ids"] == ["stable_project_facts"]


def test_new_advisory_label_needs_no_enum_and_preserves_raw_refinement_evidence(
    tmp_path: Path,
) -> None:
    catalog = compile_semantic_catalog(load_memory_blocks(tmp_path, "alpha"))
    label = DynamicLabel(
        label_id="release-oddity",
        label="release oddity",
        semantic_memberships=("novel-release-class",),
        scope="task",
        workspace_id="alpha",
        task_id="release-task",
        created_at="2026-07-14T00:00:00+00:00",
    )
    case = RefinementCase(
        case_id="release-oddity-evidence",
        case_type="novel_concept",
        failure_stage="selection",
        query="cobalt release marker",
        decision_id="decision-release-oddity",
        task_signature="release:classification",
        action_trace_id="action-release-oddity",
        evidence_refs=(_evidence("raw-release-log-42"),),
        reason="novel release concept",
        selected_descriptor_ids=("release-oddity",),
    )

    extended = catalog.with_dynamic_label(label).with_refinement_case(case)

    assert extended.revision_id != catalog.revision_id
    assert extended.dynamic_labels[0].semantic_memberships == ("novel-release-class",)
    assert extended.refinement_cases[0].evidence_refs == (_evidence("raw-release-log-42"),)


def test_catalog_session_pin_and_rollback_restore_routing_without_canonical_mutation(
    tmp_path: Path,
) -> None:
    blocks = load_memory_blocks(tmp_path, "alpha")
    initial_snapshot = catalog_snapshot_from_blocks(blocks)
    initial_bundle = baseline_memory_policy_bundle(initial_snapshot)
    store = ServicePolicyBundleStore(
        tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    store.ensure_initialized(initial_bundle, initial_snapshot)
    pinned = store.pin_session("task-session")

    changed_blocks = [
        replace(block, tags=(*block.tags, "rollback-probe"))
        if block.block_id == "stable_project_facts"
        else block
        for block in blocks
    ]
    changed_snapshot = catalog_snapshot_from_blocks(changed_blocks)
    changed_bundle = baseline_memory_policy_bundle(changed_snapshot)
    activated = store.activate(
        changed_bundle,
        activation_id="activate-changed-catalog",
        catalog_snapshot=changed_snapshot,
    )
    assert activated.status == "activated"
    assert store.pin_session("task-session") == pinned

    rolled_back = store.activate(
        initial_bundle,
        activation_id="rollback-initial-catalog",
        catalog_snapshot=initial_snapshot,
    )

    assert rolled_back.status == "activated"
    assert store.active_bundle() == initial_bundle
    assert store.snapshot_for(initial_bundle) == initial_snapshot
