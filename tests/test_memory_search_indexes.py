from dataclasses import replace
from pathlib import Path

from lethyx import MemoryClient
from lethyx.contracts import (
    DerivationReceipt,
    EvidenceRef,
    MemoryBlockRecord,
    OriginRef,
    OriginSegment,
    RecordBlockMembership,
    TransferEnvelope,
    TurnChildClaim,
)
from lethyx.memory.backends.sqlite_index import (
    SQLITE_INDEX_SCHEMA_VERSION,
    BackendSQLiteIndex,
)


def _evidence(name: str) -> EvidenceRef:
    return EvidenceRef(
        ref_type="test_run",
        ref_id=name,
        produced_by="verifier",
        producer_id=f"verifier:index:{name}",
    )


def _rich_record(block_id: str = "indexed-record") -> MemoryBlockRecord:
    raw = OriginSegment(
        claim_id="claim-raw",
        origin=OriginRef(
            channel="tool_result",
            principal_id="pytest",
            integrity_class="observed",
            origin_id="tool:pytest:raw-1",
        ),
        integrity_ceiling="observed",
        independence_domain="pytest:independent:tool",
        evidence_refs=(_evidence("raw-source"),),
    )
    claim = TurnChildClaim(
        claim_type="derived_fact",
        source_role="assistant",
        content="The endpoint is implemented by ReleaseController.publish.",
        parent_session_id="session-1",
        origin_segments=(raw,),
        derivation_receipt=DerivationReceipt(
            derived_id="derived:release-endpoint",
            source_segments=(raw,),
            transformation="extract",
            integrity_ceiling="observed",
        ),
    )
    return MemoryBlockRecord(
        block_id=block_id,
        revision_id=f"revision:{block_id}",
        recorded_at="2026-01-01T00:00:00Z",
        content="ReleaseController.publish serves POST /v1/releases with release.channel.",
        title="Release publishing endpoint",
        kind="artifact_claim",
        status="candidate",
        retention="durable",
        delivery_policy="on_demand",
        temperature="warm",
        authority="advisory",
        trust_level="agent_reported",
        scope="workspace",
        verification_status="unverified",
        promotion_decision="not_requested",
        workspace_id="ws",
        run_id="run-1",
        block_membership=RecordBlockMembership(
            primary_block_id="artifact_claims",
            catalog_revision_id="catalog:test",
        ),
        repository_id="repo:lethyx",
        artifact_id="wheel",
        artifact_revision="sha256:ABCDEF1234",
        child_claims=(claim,),
        metadata={
            "package_id": "lethyx",
            "component_id": "release",
            "dependencies": ["sqlite"],
            "configuration": ["release-mode"],
            "commands": ["uv run pytest"],
            "version": "2.4.0",
            "temporal_state": "current",
            "paths": ["src/lethyx/release/controller.py"],
            "symbols": ["ReleaseController.publish"],
            "entities": ["ReleaseController"],
            "config_keys": ["release.channel"],
            "endpoints": ["POST /v1/releases"],
            "artifact_hashes": ["sha256:ABCDEF1234"],
            "checkpoint_source_range_refs": ["events:100-120"],
            "fold_source_range_refs": ["events:80-99"],
        },
    )


def test_schema_has_versioned_exact_fts_temporal_and_provenance_indexes(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    try:
        assert index._conn.execute(
            "SELECT value FROM schema_metadata WHERE key='schema_version'"
        ).fetchone() == (SQLITE_INDEX_SCHEMA_VERSION,)
        tables = {
            row[0]
            for row in index._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "memory_exact_terms",
            "memory_fts",
            "memory_temporal_index",
            "memory_provenance_links",
        }.issubset(tables)
    finally:
        index.close()
    reopened = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    reopened.close()


def test_write_builds_structure_exact_facets_and_reverse_provenance(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    record = _rich_record()
    envelope = TransferEnvelope(
        envelope_id="transfer-release",
        evidence_refs=(_evidence("transfer"),),
        status="accepted",
        source_scope="workspace:lethyx",
        target_scope="workspace:release-fixture",
        source_task_signature="repair:python",
        target_task_signature="release:fixture",
        qualified_memory_ids=(record.block_id,),
        verified_outcome_ids=("outcome-a", "outcome-b"),
        independence_domains=("project:lethyx", "project:release-fixture"),
        source_workflow="repair-python",
        target_workflow="release-fixture",
        source_project_id="lethyx",
        target_project_id="release-fixture",
        source_host="hermes",
        target_host="codex",
        source_profile="hermes:deep",
        target_profile="codex:deep",
        source_stages=("repair", "verify"),
        target_stages=("package", "verify"),
        prerequisites=("typed-tests",),
        contraindications=("untrusted-origin",),
        exclusions=("production-erase",),
        support_count=2,
        source_revision_id="lethyx@abc123",
        valid_from="2026-01-01T00:00:00Z",
        expires_at="2027-01-01T00:00:00Z",
        drift_fingerprint="sha256:transfer-release",
    )
    try:
        index.retain_block(record)
        index.refresh_memory_indexes(record.block_id, transfer_envelope=envelope)

        terms = index.exact_index_terms(record.block_id)
        assert terms["path"] == ("src/lethyx/release/controller.py",)
        assert terms["package"] == ("lethyx",)
        assert terms["component"] == ("release",)
        assert terms["symbol"] == ("ReleaseController.publish",)
        assert terms["entity"] == ("ReleaseController",)
        assert terms["config_key"] == ("release.channel",)
        assert terms["dependency"] == ("sqlite",)
        assert terms["configuration"] == ("release-mode",)
        assert terms["command"] == ("uv run pytest",)
        assert terms["version"] == ("2.4.0",)
        assert terms["temporal_state"] == ("current",)
        assert terms["endpoint"] == ("POST /v1/releases",)
        assert terms["artifact_hash"] == ("sha256:ABCDEF1234",)
        assert terms["origin_channel"] == ("tool_result",)
        assert terms["independence_domain"] == ("pytest:independent:tool",)
        assert terms["checkpoint_source_range"] == ("events:100-120",)
        assert terms["fold_source_range"] == ("events:80-99",)
        assert terms["transfer_source_scope"] == ("workspace:lethyx",)
        assert terms["transfer_target_scope"] == ("workspace:release-fixture",)

        nodes = index.project_structure_nodes(record.block_id)
        assert [node.metadata["structure_kind"] for node in nodes] == [
            "repository",
            "package",
            "component",
            "file",
            "symbol",
        ]
        assert [node.metadata["structure_value"] for node in nodes] == [
            "repo:lethyx",
            "lethyx",
            "release",
            "src/lethyx/release/controller.py",
            "ReleaseController.publish",
        ]
        assert all(
            child.metadata["parent_id"] == parent.node_id for parent, child in zip(nodes, nodes[1:])
        )
        facet_nodes = index.select_index_nodes(
            workspace_id="ws",
            node_type="exact_facet",
            limit=100,
        )
        assert {node.metadata["term_type"] for node in facet_nodes} == {
            "command",
            "config_key",
            "configuration",
            "dependency",
            "endpoint",
            "entity",
            "temporal_state",
            "version",
        }
        assert all(
            node.metadata["term_type"]
            not in {
                "artifact_hash",
                "checkpoint_source_range",
                "fold_source_range",
                "independence_domain",
                "origin_channel",
                "raw_source_ref",
                "transfer_source_scope",
                "transfer_target_scope",
            }
            for node in facet_nodes
        )
        assert index.reverse_provenance_sources("derived:release-endpoint") == (
            "raw-source",
            "tool:pytest:raw-1",
        )

        exact = index.recall(
            {
                "query": "",
                "workspace_id": "ws",
                "run_id": "run-1",
                "scope_chain": ["workspace:ws"],
                "symbols": ["ReleaseController.publish"],
                "artifact_hashes": ["sha256:ABCDEF1234"],
                "include_candidates": True,
            }
        )
        assert [item.block_id for item in exact] == [record.block_id]
        missing = index.recall(
            {
                "query": "",
                "workspace_id": "ws",
                "run_id": "run-1",
                "scope_chain": ["workspace:ws"],
                "symbols": ["ReleaseController.missing"],
                "include_candidates": True,
            }
        )
        assert missing == []
    finally:
        index.close()


def test_fts5_bm25_is_primary_and_recent_substring_is_failure_only_fallback(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    try:
        index.retain_block(_rich_record("fts-hit"))
        index.retain_block(
            replace(
                _rich_record("fts-miss"),
                content="Unrelated continuity note.",
                title="Continuity",
                metadata={},
                child_claims=(),
            )
        )
        [hit] = index.recall(
            {
                "query": "ReleaseController publish",
                "workspace_id": "ws",
                "run_id": "run-1",
                "scope_chain": ["workspace:ws"],
                "include_candidates": True,
            }
        )
        assert hit.block_id == "fts-hit"
        assert hit.metadata["retrieval_channel"] == "sqlite_fts5_bm25"
        assert isinstance(hit.metadata["retrieval_bm25"], float)

        index._conn.execute("DROP TABLE memory_fts")
        [fallback] = index.recall(
            {
                "query": "continuity",
                "workspace_id": "ws",
                "run_id": "run-1",
                "scope_chain": ["workspace:ws"],
                "include_candidates": True,
            }
        )
        assert fallback.block_id == "fts-miss"
        assert fallback.metadata["retrieval_channel"] == "recent_substring_fallback"
        assert fallback.metadata["retrieval_scan_limit"] <= 128
    finally:
        index.close()


def test_logical_block_scope_filters_fts_without_turning_into_exact_record_fetch(
    tmp_path: Path,
) -> None:
    index = BackendSQLiteIndex(tmp_path / "scoped.sqlite", backend_id="test", workspace_id="ws")

    def scoped_record(block_id: str, *, title: str, content: str) -> MemoryBlockRecord:
        record = replace(
            _rich_record(block_id),
            title=title,
            content=content,
            revision_id=f"revision:{block_id}",
            metadata={},
            child_claims=(),
        )
        membership = RecordBlockMembership(
            primary_block_id="workspace_episodes",
            catalog_revision_id="catalog:test",
            record_id=record.block_id,
            record_revision_id=record.revision_id,
            workspace_id=record.workspace_id,
            valid_from=record.recorded_at,
        )
        return replace(record, block_membership=membership)

    try:
        index.retain_block(
            scoped_record(
                "a_irrelevant",
                title="Unrelated",
                content="material about an unrelated topic",
            )
        )
        index.retain_block(
            scoped_record(
                "z_relevant",
                title="Requested semantic answer",
                content="the requested semantic answer",
            )
        )

        recalled = index.recall(
            {
                "query": "requested semantic answer",
                "workspace_id": "ws",
                "run_id": "run-1",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["workspace_episodes"],
                "include_candidates": True,
                "limit": 1,
            }
        )
    finally:
        index.close()

    assert [record.block_id for record in recalled] == ["z_relevant"]


def test_temporal_index_serves_current_and_historical_revisions(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "temporal.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    try:
        original_result = client.remember(
            "The release endpoint uses the legacy writer.",
            title="Release writer",
            kind="episode",
            requested_block_id="workspace_episodes",
            metadata={"symbols": ["ReleaseController.publish"]},
            _recorded_at="2026-07-14T18:00:00Z",
        )
        original = index.get_block(original_result.memory_id)
        assert original is not None
        corrected_result = client.correction(
            original.block_id,
            "The release endpoint uses the verified writer.",
            reason="verified implementation",
            requested_block_id="workspace_episodes",
        )
        corrected = index.get_block(str(corrected_result["block_id"]))
        assert corrected is not None

        entries = index.temporal_index_entries(original.canonical_key)
        assert [(item["block_id"], item["is_current"]) for item in entries] == [
            (original.block_id, False),
            (corrected.block_id, True),
        ]
        assert entries[0]["valid_to"] == corrected.valid_from

        current = index.recall(
            {
                "query": "verified writer",
                "workspace_id": "ws",
                "include_candidates": True,
                "source_horizon": "archive",
                "scope_chain": ["workspace:ws"],
            }
        )
        assert [item.block_id for item in current] == [corrected.block_id]
        historical = index.recall(
            {
                "query": "legacy writer",
                "workspace_id": "ws",
                "include_candidates": True,
                "source_horizon": "archive",
                "scope_chain": ["workspace:ws"],
                "temporal_mode": "historical",
                "revision_id": original.revision_id,
            }
        )
        assert [item.block_id for item in historical] == [original.block_id]
    finally:
        index.close()
