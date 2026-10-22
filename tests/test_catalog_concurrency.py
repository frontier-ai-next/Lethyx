from concurrent.futures import ThreadPoolExecutor

from um_mem.memory.blocks.models import MemoryBlock
from um_mem.memory.catalog_store import WorkspaceCatalogStore


def _block(block_id: str, parent_id: str) -> MemoryBlock:
    return MemoryBlock(
        block_id=block_id,
        block_type="semantic_leaf",
        temperature="warm",
        scope="workspace",
        authority="advisory",
        delivery_policy="on_demand",
        retention="durable",
        name=block_id,
        parent_id=parent_id,
        agent_writable=True,
        index_only=False,
        owner="test",
        producer="test",
        consumers=("memory_agent",),
        recall_path="hybrid_recall",
        lifecycle_policy={
            "initial_status": "active",
            "mutable": True,
            "terminal_statuses": ("archived",),
        },
        isolation_policy={"dimension": "workspace", "cross_workspace": False},
        token_budget=128,
    )


def test_concurrent_catalog_appends_allocate_unique_revisions(tmp_path) -> None:
    stores_count = 8
    root_store = WorkspaceCatalogStore(tmp_path, "ws", "default")
    try:
        root = root_store.list_blocks()[0].block
    finally:
        root_store.close()

    def create(index: int) -> None:
        store = WorkspaceCatalogStore(tmp_path, "ws", "default")
        try:
            store.create_block(
                _block(f"leaf-{index}", root.block_id),
                backend_partition_id=f"partition-{index}",
                actor="test",
                reason="concurrent append",
            )
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=stores_count) as executor:
        list(executor.map(create, range(stores_count)))

    reopened = WorkspaceCatalogStore(tmp_path, "ws", "default")
    try:
        rows = reopened._conn.execute(
            "SELECT previous_revision, next_revision FROM catalog_mutations "
            "WHERE operation='create_block' ORDER BY next_revision"
        ).fetchall()
        assert rows == [(index, index + 1) for index in range(1, stores_count + 1)]
        assert reopened.catalog_revision() == stores_count + 1
    finally:
        reopened.close()
