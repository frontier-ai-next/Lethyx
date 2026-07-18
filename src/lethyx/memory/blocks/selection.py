"""Phase-aware block selection and legacy recall-policy compilation."""

from lethyx.memory.blocks.models import MemoryBlock


def _matches(values: object, candidate: str) -> bool:
    if not values:
        return True
    if isinstance(values, str):
        values = [values]
    if isinstance(values, (list, tuple, set)):
        return "*" in values or candidate in values
    return True


def select_blocks_for_phase(
    blocks: list[MemoryBlock],
    phase_id: str,
    workspace_id: str = "",
    agent_kind: str = "generic_agent",
    token_budget: int = 4000,
) -> list[MemoryBlock]:
    selected: list[MemoryBlock] = []
    spent = 0
    for block in blocks:
        applies = block.applies_to or {}
        if not _matches(applies.get("phases"), phase_id):
            continue
        if not _matches(applies.get("workspaces"), workspace_id or "*"):
            continue
        if not _matches(applies.get("agents"), agent_kind):
            continue
        if spent + block.token_budget > token_budget and block.tier != "always_on":
            continue
        selected.append(block)
        spent += max(0, block.token_budget)
    return selected


def compile_block_recall_policy(blocks: list[MemoryBlock]) -> dict[str, list[dict[str, object]]]:
    policy: dict[str, list[dict[str, object]]] = {
        "always_on": [],
        "hot": [],
        "warm_search": [],
        "graph": [],
    }
    for block in blocks:
        row = {
            "store": block.store,
            "tier": block.tier,
            "tags": list(block.tags),
            "block_id": block.block_id,
        }
        if block.tier == "always_on":
            policy["always_on"].append(row)
        elif block.tier == "hot":
            policy["hot"].append(row)
        else:
            row.update(block.retrieval_policy)
            policy["warm_search"].append(row)
        if block.graph_policy:
            policy["graph"].append({"block_id": block.block_id, **block.graph_policy})
    return policy
