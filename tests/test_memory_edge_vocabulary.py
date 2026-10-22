from um_mem.memory.backends.contracts import ALLOWED_EDGE_TYPES, EdgeType, MemoryEdgeRecord


def test_graph_edge_vocabulary_is_allowed_by_backend_contract() -> None:
    expected = {
        EdgeType.SUPPORTS,
        EdgeType.VERIFIED_BY,
        EdgeType.CONTRADICTS,
        EdgeType.SUPERSEDES,
    }

    assert expected.issubset(ALLOWED_EDGE_TYPES)
    for edge_type in expected:
        record = MemoryEdgeRecord(src_id="a", dst_id="b", edge_type=edge_type)
        assert record.edge_type == edge_type

    assert "superseded_by" not in ALLOWED_EDGE_TYPES
