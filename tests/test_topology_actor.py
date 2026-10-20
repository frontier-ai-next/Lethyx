"""Host provisioning is not attributed to an LLM in the immutable mutation log."""

import json
import sqlite3
from pathlib import Path

import pytest

from lethyx import MemoryClient
from lethyx.memory.catalog_store import WorkspaceCatalogStore


@pytest.mark.parametrize("actor", ["llm", "system"])
def test_public_topology_actor_is_retained_for_create_amend_move_archive(
    tmp_path: Path,
    actor: str,
) -> None:
    client = MemoryClient(tmp_path, workspace_id="actor-test", topology_actor=actor)
    root = client.block_map()[0]
    left = client.create_index_block(
        name="Left", synopsis="First", parent_id=root["block_id"], reason="index"
    )
    right = client.create_index_block(
        name="Right", synopsis="Second", parent_id=root["block_id"], reason="index"
    )
    leaf = client.create_leaf_block(
        name="Result", synopsis="Fact", parent_id=left["block_id"], reason="create"
    )
    amended = client.amend_block(
        leaf["block_id"], expected_revision=leaf["revision"], synopsis="Updated", reason="amend"
    )
    moved = client.move_block(
        leaf["block_id"],
        expected_revision=amended["revision"],
        parent_id=right["block_id"],
        reason="move",
    )
    client.archive_block(leaf["block_id"], expected_revision=moved["revision"], reason="archive")
    store = WorkspaceCatalogStore(tmp_path, "actor-test", "default")
    try:
        with sqlite3.connect(store.path) as connection:
            rows = connection.execute(
                "SELECT actor, affected_ids_json FROM catalog_mutations"
            ).fetchall()
        actors = [who for who, ids in rows if leaf["block_id"] in json.loads(ids)]
        assert actors == [actor] * 4
    finally:
        store.close()


def test_topology_actor_rejects_an_unrecognized_host_label(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="topology_actor"):
        MemoryClient(tmp_path, topology_actor="unknown-agent")
