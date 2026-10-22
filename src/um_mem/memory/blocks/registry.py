"""Load built-in and workspace memory block definitions."""

import json
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import yaml

from um_mem.contracts.memory import (
    DeliveryPolicy,
    MemoryRetention,
    MemoryScope,
    MemoryTemperature,
    _closed_value,
)
from um_mem.memory.blocks.models import MemoryBlock
from um_mem.memory.catalog_store import TopologyBlockState, WorkspaceCatalogStore

_YAML_LOADER: "type[yaml.CSafeLoader | yaml.SafeLoader]"
try:
    _YAML_LOADER = yaml.CSafeLoader
except AttributeError:  # pragma: no cover - libyaml absent
    _YAML_LOADER = yaml.SafeLoader


_BOUND_CATALOG: ContextVar[tuple[Path, str, str, tuple[dict[str, Any], ...]] | None] = ContextVar(
    "um_mem_bound_memory_block_catalog", default=None
)
_DYNAMIC_BLOCK_OWNER = "agent"
_DYNAMIC_BLOCK_SCOPES = frozenset(
    {"workspace", "project", "session", "task", "subtask", "user_global"}
)
_WORKSPACE_TOPOLOGY_BLOCK_TYPES = frozenset(
    {
        "workspace_profile_root",
        "workspace_dynamic",
        "workspace_dynamic_index",
    }
)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values: Iterable[object]
    if isinstance(value, str):
        values = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        values = (value,)
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_block(raw: dict[str, Any]) -> MemoryBlock | None:
    if not isinstance(raw, dict) or not raw.get("block_id"):
        return None
    unknown_fields = sorted(set(raw).difference(MemoryBlock.__dataclass_fields__))
    if unknown_fields:
        raise ValueError("unknown memory block fields: " + ", ".join(unknown_fields))
    retrieval_policy = dict(raw.get("retrieval_policy") or {})
    unknown_policy_fields = sorted(
        set(retrieval_policy).difference({"mode", "limit", "require_query_seed"})
    )
    if unknown_policy_fields:
        raise ValueError("unknown retrieval_policy fields: " + ", ".join(unknown_policy_fields))
    return MemoryBlock(
        block_id=str(raw.get("block_id") or ""),
        block_type=str(raw.get("block_type") or "generic"),
        temperature=cast(
            MemoryTemperature,
            _closed_value(
                raw.get("temperature") or "warm", field_name="temperature", shape=MemoryTemperature
            ),
        ),
        scope=cast(
            MemoryScope,
            _closed_value(raw.get("scope") or "workspace", field_name="scope", shape=MemoryScope),
        ),
        authority=str(raw.get("authority") or "advisory"),
        delivery_policy=cast(
            DeliveryPolicy,
            _closed_value(
                raw.get("delivery_policy") or "on_demand",
                field_name="delivery_policy",
                shape=DeliveryPolicy,
            ),
        ),
        retention=cast(
            MemoryRetention,
            _closed_value(
                raw.get("retention") or "durable", field_name="retention", shape=MemoryRetention
            ),
        ),
        name=str(raw.get("name") or raw.get("block_id") or ""),
        tags=_string_tuple(raw.get("tags")),
        semantic_memberships=_string_tuple(raw.get("semantic_memberships")),
        guidance=dict(raw.get("guidance") or {}),
        metadata=dict(raw.get("metadata") or {}),
        parent_id=str(raw.get("parent_id") or ""),
        child_ids=_string_tuple(raw.get("child_ids")),
        index_only=_bool(raw.get("index_only")),
        agent_writable=(
            True if raw.get("agent_writable") is None else _bool(raw.get("agent_writable"))
        ),
        owner=str(raw.get("owner") or "UM-MEM"),
        producer=str(raw.get("producer") or ""),
        consumers=_string_tuple(raw.get("consumers")),
        recall_path=str(raw.get("recall_path") or ""),
        lifecycle_policy=dict(raw.get("lifecycle_policy") or {}),
        isolation_policy=dict(raw.get("isolation_policy") or {}),
        applies_to=dict(raw.get("applies_to") or {}),
        retrieval_policy=retrieval_policy,
        graph_policy=dict(raw.get("graph_policy") or {}),
        promotion_policy=dict(raw.get("promotion_policy") or {}),
        token_budget=int(1000 if raw.get("token_budget") is None else raw["token_budget"]),
        status_policy=dict(raw.get("status_policy") or {}),
    )


def _load_yaml(path: Path, *, strict: bool = False) -> list[MemoryBlock]:
    if not path.is_file():
        return []
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_YAML_LOADER) or {}
    except (OSError, yaml.YAMLError) as exc:
        if strict:
            raise ValueError(f"memory block catalog is invalid: {path}") from exc
        return []
    raw_blocks = data.get("blocks") if isinstance(data, dict) else data
    if not isinstance(raw_blocks, list):
        if strict:
            raise ValueError(f"memory block catalog must contain a blocks list: {path}")
        return []
    blocks: list[MemoryBlock] = []
    for raw in raw_blocks:
        try:
            block = _coerce_block(raw)
        except (TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"memory block catalog contains an invalid block: {path}") from exc
            continue
        if block is not None:
            blocks.append(block)
    return blocks


def default_blocks_path() -> Path:
    return Path(__file__).with_name("default_blocks.yaml")


def taxonomy_path() -> Path:
    return Path(__file__).with_name("taxonomy.yaml")


def load_workspace_topology(
    repo_root: str | Path,
    workspace_id: str,
    *,
    profile_id: str = "default",
    include_archived: bool = False,
) -> list[TopologyBlockState]:
    store = WorkspaceCatalogStore(repo_root, workspace_id, profile_id)
    try:
        return store.list_blocks(include_archived=include_archived)
    finally:
        store.close()


def _validate_dynamic_parent(
    blocks: list[MemoryBlock],
    *,
    block_id: str,
    parent_id: str,
) -> None:
    by_id = {block.block_id: block for block in blocks}
    parent = by_id.get(parent_id)
    if parent is None:
        raise ValueError("dynamic memory block parent is not active in this workspace")
    if parent.block_id == block_id:
        raise ValueError("dynamic memory block hierarchy would create a cycle")
    if not parent.index_only:
        raise ValueError("dynamic memory blocks require an index-only parent")


def create_workspace_block(
    repo_root: str | Path,
    workspace_id: str,
    *,
    profile_id: str = "default",
    name: str,
    synopsis: str,
    parent_id: str,
    reason: str,
    actor: str = "llm",
    tags: list[str] | tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
    scope: str = "workspace",
    index_only: bool = False,
    token_budget: int = 1000,
    protected: bool = False,
    lifecycle_expires_at: str = "",
) -> dict[str, Any]:
    identifier = str(uuid.uuid4())
    display_name = str(name or "").strip()
    synopsis_value = str(synopsis or "").strip()
    parent = str(parent_id or "").strip()
    mutation_reason = str(reason or "").strip()
    block_scope = str(scope or "").strip().casefold()
    if not display_name or not synopsis_value or not parent or not mutation_reason:
        raise ValueError("dynamic memory block name, synopsis, parent_id, and reason are required")
    if isinstance(token_budget, bool) or int(token_budget) < 0:
        raise ValueError("dynamic memory block token_budget must be non-negative")
    if not isinstance(protected, bool):
        raise ValueError("dynamic memory block protected flag must be boolean")
    if block_scope not in _DYNAMIC_BLOCK_SCOPES:
        raise ValueError("dynamic memory block scope is unsupported")
    metadata_value = dict(metadata or {})
    json.dumps(metadata_value, ensure_ascii=True, sort_keys=True)
    store = WorkspaceCatalogStore(repo_root, workspace_id, profile_id)
    try:
        topology = store.list_blocks()
        catalog = [state.block for state in topology]
        _validate_dynamic_parent(catalog, block_id=identifier, parent_id=parent)
        lifecycle_policy = {
            "initial_status": "candidate",
            "mutable": True,
            "terminal_statuses": ("superseded", "deprecated", "retracted"),
        }
        if str(lifecycle_expires_at or "").strip():
            lifecycle_policy["expires_at"] = str(lifecycle_expires_at).strip()
        block = MemoryBlock(
            block_id=identifier,
            block_type="workspace_dynamic_index" if index_only else "workspace_dynamic",
            temperature="warm",
            scope=cast(MemoryScope, block_scope),
            authority="advisory",
            delivery_policy="on_demand",
            retention="durable",
            name=display_name,
            tags=tuple(tags),
            guidance={
                "synopsis": synopsis_value,
                "use_when": (),
                "avoid_when": (),
            },
            metadata=metadata_value,
            parent_id=parent,
            index_only=bool(index_only),
            agent_writable=not bool(index_only),
            owner=_DYNAMIC_BLOCK_OWNER,
            producer="agent",
            consumers=("memory_agent", "context_packer"),
            recall_path="hybrid_recall",
            lifecycle_policy=lifecycle_policy,
            isolation_policy={
                "dimension": block_scope,
                "cross_workspace": block_scope == "user_global",
            },
            retrieval_policy={"mode": "hybrid", "limit": 6, "require_query_seed": True},
            token_budget=int(token_budget),
        )
        candidate = [*catalog, block]
        validate_memory_block_tree(candidate)
        validate_leaf_block_contracts(candidate)
        state = store.create_block(
            block,
            backend_partition_id="" if index_only else str(uuid.uuid4()),
            actor=actor,
            reason=mutation_reason,
            protected=protected,
        )
        return {
            "ok": True,
            "saved": True,
            **state.to_mapping(),
            "catalog_revision": store.catalog_revision(),
            "block": asdict(block),
        }
    finally:
        store.close()


def amend_workspace_block(
    repo_root: str | Path,
    workspace_id: str,
    block_id: str,
    *,
    expected_revision: int,
    profile_id: str = "default",
    name: str | None = None,
    synopsis: str | None = None,
    parent_id: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    metadata_patch: Mapping[str, Any] | None = None,
    token_budget: int | None = None,
    archived: bool | None = None,
    reason: str,
    actor: str = "llm",
    allow_protected_session: bool = False,
    expected_session_id: str = "",
    lifecycle_expires_at: str | None = None,
) -> dict[str, Any]:
    identifier = str(block_id or "").strip()
    mutation_reason = str(reason or "").strip()
    if not identifier or type(expected_revision) is not int or not mutation_reason:
        raise ValueError("amend_block requires block_id, integer expected_revision, and reason")
    store = WorkspaceCatalogStore(repo_root, workspace_id, profile_id)
    try:
        state = store.get_block(identifier)
        if state is None:
            raise ValueError(f"dynamic memory block {identifier!r} was not found in this profile")
        protected_session = (
            bool(allow_protected_session)
            and str(state.block.scope).strip().casefold() == "session"
            and str(state.block.metadata.get("umbrella_memory_session_id") or "").strip()
            == str(expected_session_id or "").strip()
            and bool(str(expected_session_id or "").strip())
        )
        if state.protected and not protected_session:
            raise ValueError("protected root cannot be amended")
        current = state.block
        metadata_value = dict(current.metadata)
        for key, value in dict(metadata_patch or {}).items():
            if value is None:
                metadata_value.pop(str(key), None)
            else:
                metadata_value[str(key)] = value
        json.dumps(metadata_value, ensure_ascii=True, sort_keys=True)
        guidance = dict(current.guidance)
        if synopsis is not None:
            synopsis_value = str(synopsis).strip()
            if not synopsis_value:
                raise ValueError("dynamic memory block synopsis cannot be empty")
            guidance["synopsis"] = synopsis_value
        display_name = current.name if name is None else str(name).strip()
        if not display_name:
            raise ValueError("dynamic memory block name cannot be empty")
        target_parent = current.parent_id if parent_id is None else str(parent_id).strip()
        if not target_parent:
            raise ValueError("dynamic memory block parent_id cannot be empty")
        budget = current.token_budget if token_budget is None else int(token_budget)
        if isinstance(token_budget, bool) or budget < 0:
            raise ValueError("dynamic memory block token_budget must be non-negative")
        updated = replace(
            current,
            name=display_name,
            guidance=guidance,
            metadata=metadata_value,
            parent_id=target_parent,
            tags=current.tags if tags is None else tuple(tags),
            token_budget=budget,
            lifecycle_policy=(
                {
                    **current.lifecycle_policy,
                    **(
                        {"expires_at": str(lifecycle_expires_at).strip()}
                        if str(lifecycle_expires_at or "").strip()
                        else {}
                    ),
                }
                if lifecycle_expires_at is not None and str(lifecycle_expires_at).strip()
                else {
                    key: value
                    for key, value in current.lifecycle_policy.items()
                    if key != "expires_at"
                }
                if lifecycle_expires_at is not None
                else current.lifecycle_policy
            ),
        )
        should_archive = state.status == "archived" if archived is None else bool(archived)
        active_states = store.list_blocks()
        active_without_current = [
            item.block for item in active_states if item.block_id != identifier
        ]
        if should_archive:
            children = [
                block.block_id for block in active_without_current if block.parent_id == identifier
            ]
            if children:
                raise ValueError("dynamic memory block with active children cannot be archived")
            incoming = [edge for edge in store.edges_for(identifier) if edge.dst_id == identifier]
            if incoming:
                raise ValueError(
                    "dynamic memory block with incoming graph edges cannot be archived"
                )
            if store.block_has_active_mask_reference(identifier):
                raise ValueError(
                    "dynamic memory block with active mask references cannot be archived"
                )
        else:
            candidate = [*active_without_current, updated]
            _validate_dynamic_parent(
                candidate,
                block_id=identifier,
                parent_id=target_parent,
            )
            validate_memory_block_tree(candidate)
            validate_leaf_block_contracts(candidate)
        next_state = store.replace_block(
            state,
            updated,
            expected_revision=expected_revision,
            status="archived" if should_archive else "active",
            actor=actor,
            reason=mutation_reason,
            operation=(
                "archive_block"
                if should_archive and state.status != "archived"
                else "revive_block"
                if not should_archive and state.status == "archived"
                else "amend_block"
            ),
            allow_protected=protected_session,
        )
        return {
            "ok": True,
            "saved": True,
            "previous_revision": state.revision,
            **next_state.to_mapping(),
            "archived": next_state.status == "archived",
            "catalog_revision": store.catalog_revision(),
            "block": asdict(updated),
        }
    finally:
        store.close()


def convert_workspace_block_to_index(
    repo_root: str | Path,
    workspace_id: str,
    block_id: str,
    *,
    expected_revision: int,
    reason: str,
    profile_id: str = "default",
    actor: str = "llm",
) -> dict[str, Any]:
    """Turn an empty leaf into an index-only parent so it can hold children.

    Conversion is deliberately limited to blocks that have never held records.
    The caller proves that separately against the selected backend; this function
    keeps the catalog mutation atomic and clears the now-unused partition id.
    """

    identifier = str(block_id or "").strip()
    mutation_reason = str(reason or "").strip()
    if not identifier or type(expected_revision) is not int or not mutation_reason:
        raise ValueError(
            "convert_block_to_index requires block_id, integer expected_revision, and reason"
        )
    store = WorkspaceCatalogStore(repo_root, workspace_id, profile_id)
    try:
        state = store.get_block(identifier)
        if state is None:
            raise ValueError(f"dynamic memory block {identifier!r} was not found in this profile")
        if state.protected:
            raise ValueError("protected root cannot be converted to an index block")
        if state.status != "active":
            raise ValueError("only an active leaf can be converted to an index block")
        if state.block.index_only:
            return {
                "ok": True,
                "saved": False,
                "already_index_only": True,
                "block_id": identifier,
                "revision": state.revision,
                "catalog_revision": store.catalog_revision(),
                "block": asdict(state.block),
            }
        if state.block.child_ids:
            raise ValueError("a block with children cannot be converted to an index block")
        updated = replace(
            state.block,
            block_type="workspace_dynamic_index",
            index_only=True,
            agent_writable=False,
        )
        active_without_current = [
            item.block for item in store.list_blocks() if item.block_id != identifier
        ]
        candidate = [*active_without_current, updated]
        validate_memory_block_tree(candidate)
        validate_leaf_block_contracts(candidate)
        converted = store.replace_block(
            state,
            updated,
            expected_revision=expected_revision,
            status="active",
            actor=actor,
            reason=mutation_reason,
            operation="convert_block_to_index",
            backend_partition_id="",
        )
        return {
            "ok": True,
            "saved": True,
            "previous_revision": state.revision,
            **converted.to_mapping(),
            "catalog_revision": store.catalog_revision(),
            "block": asdict(updated),
        }
    finally:
        store.close()


def repair_protected_workspace_block_parent(
    repo_root: str | Path,
    workspace_id: str,
    block_id: str,
    *,
    expected_revision: int,
    expected_parent_id: str,
    parent_id: str,
    expected_metadata: Mapping[str, Any],
    reason: str,
    profile_id: str = "default",
) -> dict[str, Any]:
    identifier = str(block_id or "").strip()
    current_parent_id = str(expected_parent_id or "").strip()
    target_parent_id = str(parent_id or "").strip()
    mutation_reason = str(reason or "").strip()
    metadata_guard = {str(key): value for key, value in dict(expected_metadata).items()}
    if (
        not identifier
        or type(expected_revision) is not int
        or not current_parent_id
        or not target_parent_id
        or current_parent_id == target_parent_id
        or not mutation_reason
        or not metadata_guard
        or any(not key.strip() for key in metadata_guard)
    ):
        raise ValueError(
            "repair_protected_block_parent requires a block, revision, distinct parents, "
            "metadata guard, and reason"
        )

    store = WorkspaceCatalogStore(repo_root, workspace_id, profile_id)
    try:
        state = store.get_block(identifier)
        if state is None:
            raise ValueError(f"dynamic memory block {identifier!r} was not found in this profile")
        if state.revision != expected_revision:
            raise ValueError("expected_revision does not match the current block revision")
        if state.status != "active":
            raise ValueError("only an active protected block can have its parent repaired")
        if not state.protected:
            raise ValueError("only a protected block can use protected parent repair")
        if state.block.parent_id != current_parent_id:
            raise ValueError("protected block parent does not match the repair guard")
        if any(
            key not in state.block.metadata or state.block.metadata[key] != value
            for key, value in metadata_guard.items()
        ):
            raise ValueError("protected block metadata does not match the repair guard")

        updated = replace(state.block, parent_id=target_parent_id)
        active_without_current = [
            item.block for item in store.list_blocks() if item.block_id != identifier
        ]
        candidate = [*active_without_current, updated]
        _validate_dynamic_parent(
            candidate,
            block_id=identifier,
            parent_id=target_parent_id,
        )
        validate_memory_block_tree(candidate)
        validate_leaf_block_contracts(candidate)
        repaired = store.replace_block(
            state,
            updated,
            expected_revision=expected_revision,
            status="active",
            actor="system",
            reason=mutation_reason,
            operation="repair_protected_block_parent",
            allow_protected=True,
        )
        return {
            "ok": True,
            "saved": True,
            "previous_revision": state.revision,
            **repaired.to_mapping(),
            "archived": False,
            "catalog_revision": store.catalog_revision(),
            "block": asdict(updated),
        }
    finally:
        store.close()


def protect_workspace_block(
    repo_root: str | Path,
    workspace_id: str,
    block_id: str,
    *,
    expected_revision: int,
    reason: str,
    profile_id: str = "default",
    actor: str = "system",
) -> dict[str, Any]:
    identifier = str(block_id or "").strip()
    protection_reason = str(reason or "").strip()
    if not identifier or type(expected_revision) is not int or not protection_reason:
        raise ValueError("protect_block requires block_id, expected_revision, and reason")
    store = WorkspaceCatalogStore(repo_root, workspace_id, profile_id)
    try:
        state = store.get_block(identifier)
        if state is None:
            raise ValueError(f"dynamic memory block {identifier!r} was not found in this profile")
        protected = store.protect_block(
            state,
            expected_revision=expected_revision,
            actor=actor,
            reason=protection_reason,
        )
        return {
            "ok": True,
            "saved": True,
            **protected.to_mapping(),
            "catalog_revision": store.catalog_revision(),
            "block": asdict(protected.block),
        }
    finally:
        store.close()


def archive_workspace_block(
    repo_root: str | Path,
    workspace_id: str,
    block_id: str,
    *,
    expected_revision: int,
    reason: str,
    profile_id: str = "default",
    actor: str = "llm",
    allow_protected_session: bool = False,
    expected_session_id: str = "",
) -> dict[str, Any]:
    archive_reason = str(reason or "").strip()
    if not archive_reason:
        raise ValueError("archive_block reason is required")
    result = amend_workspace_block(
        repo_root,
        workspace_id,
        block_id,
        expected_revision=expected_revision,
        profile_id=profile_id,
        metadata_patch={"archive_reason": archive_reason},
        archived=True,
        reason=archive_reason,
        actor=actor,
        allow_protected_session=allow_protected_session,
        expected_session_id=expected_session_id,
    )
    result["reason"] = archive_reason
    return result


def archive_workspace_session_block(
    repo_root: str | Path,
    workspace_id: str,
    block_id: str,
    *,
    expected_revision: int,
    session_id: str,
    reason: str,
    profile_id: str = "default",
    actor: str = "system",
) -> dict[str, Any]:
    """Archive one block owned by an explicitly identified chat session.

    Protected scaffold blocks remain protected for ordinary operators.  The
    chat-forget transaction is the one narrow lifecycle transition allowed to
    archive them, and only after the caller proves the block's session
    ownership through this typed entry point.
    """

    selected_session = str(session_id or "").strip()
    if not selected_session:
        raise ValueError("session_id is required for session block archival")
    return archive_workspace_block(
        repo_root,
        workspace_id,
        block_id,
        expected_revision=expected_revision,
        reason=reason,
        profile_id=profile_id,
        actor=actor,
        allow_protected_session=True,
        expected_session_id=selected_session,
    )


def validate_memory_block_tree(blocks: list[MemoryBlock]) -> None:
    """Fail fast when logical block containment is not a valid directed tree.

    ``parent_id`` is the canonical ownership link. ``child_ids`` is an optional
    materialized index, but when it is present it must agree with the child's
    parent.  This permits a workspace to add a child beneath a built-in parent
    without replacing the parent's whole child list, while still rejecting
    dangling references, conflicting parents, and cycles.
    """

    by_id = {block.block_id: block for block in blocks}
    issues: list[str] = []
    adjacency: dict[str, set[str]] = {block_id: set() for block_id in by_id}

    for block in blocks:
        if block.parent_id:
            if block.parent_id not in by_id:
                issues.append(
                    f"block {block.block_id!r} references missing parent {block.parent_id!r}"
                )
            else:
                adjacency[block.parent_id].add(block.block_id)
        for child_id in block.child_ids:
            child = by_id.get(child_id)
            if child is None:
                issues.append(f"block {block.block_id!r} references missing child {child_id!r}")
                continue
            if child.parent_id and child.parent_id != block.block_id:
                issues.append(
                    f"block {child_id!r} declares parent {child.parent_id!r}, "
                    f"but {block.block_id!r} lists it as a child"
                )
            adjacency[block.block_id].add(child_id)
        for related_id in block.related_block_ids:
            if related_id == block.block_id:
                issues.append(f"block {block.block_id!r} cannot relate to itself")
            elif related_id not in by_id:
                issues.append(
                    f"block {block.block_id!r} references missing related block {related_id!r}"
                )

    if issues:
        raise ValueError("invalid memory block hierarchy: " + "; ".join(issues))

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(block_id: str, path: tuple[str, ...]) -> None:
        if block_id in visiting:
            cycle_start = path.index(block_id) if block_id in path else 0
            cycle = (*path[cycle_start:], block_id)
            raise ValueError(
                "invalid memory block hierarchy: cycle detected: " + " -> ".join(cycle)
            )
        if block_id in visited:
            return
        visiting.add(block_id)
        for child_id in sorted(adjacency[block_id]):
            visit(child_id, (*path, block_id))
        visiting.remove(block_id)
        visited.add(block_id)

    for block_id in sorted(by_id):
        visit(block_id, ())


def validate_leaf_block_contracts(blocks: list[MemoryBlock]) -> None:
    """Require an explicit operational contract for every storable leaf."""

    parent_ids = {block.parent_id for block in blocks if block.parent_id}
    parent_ids.update(block.block_id for block in blocks if block.child_ids)
    leaf_blocks = [block for block in blocks if block.block_id not in parent_ids]
    issues: list[str] = []
    for block in leaf_blocks:
        missing: list[str] = []
        if not block.producer:
            missing.append("producer")
        if not block.consumers:
            missing.append("consumers")
        if not block.recall_path:
            missing.append("recall_path")
        if not block.lifecycle_policy:
            missing.append("lifecycle_policy")
        if not block.isolation_policy:
            missing.append("isolation_policy")
        if missing:
            issues.append(f"{block.block_id}: {', '.join(missing)}")
    if issues:
        raise ValueError("leaf block operational contract is incomplete: " + "; ".join(issues))


def validate_taxonomy_block_ids(blocks: list[MemoryBlock]) -> None:
    """Keep packaged taxonomy level identifiers aligned with logical blocks."""

    path = taxonomy_path()
    if not path.is_file():
        return
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_YAML_LOADER) or {}
    levels = data.get("levels") if isinstance(data, dict) else {}
    if not isinstance(levels, dict):
        raise ValueError("invalid memory block taxonomy: levels must be a mapping")
    known = {block.block_id for block in blocks}
    missing = sorted(str(level_id) for level_id in levels if str(level_id) not in known)
    if missing:
        raise ValueError(
            "invalid memory block taxonomy: unknown logical block ids: " + ", ".join(missing)
        )


def catalog_snapshot_payload(
    blocks: list[MemoryBlock] | tuple[MemoryBlock, ...],
) -> list[dict[str, Any]]:
    """Return the JSON-safe immutable value used by the service catalog pin."""

    return [asdict(block) for block in blocks]


def catalog_snapshot_blocks(payload: object) -> list[MemoryBlock]:
    """Rehydrate a strict service-owned catalog snapshot."""

    if not isinstance(payload, (list, tuple)):
        raise ValueError("memory block catalog snapshot must be a list")
    blocks: list[MemoryBlock] = []
    for raw in payload:
        if not isinstance(raw, Mapping):
            raise ValueError("memory block catalog snapshot contains an invalid block")
        try:
            block = MemoryBlock(**dict(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError("memory block catalog snapshot contains an invalid block") from exc
        if not block.block_id:
            raise ValueError("memory block catalog snapshot contains an unnamed block")
        blocks.append(block)
    validate_memory_block_tree(blocks)
    validate_leaf_block_contracts(blocks)
    validate_taxonomy_block_ids(blocks)
    return blocks


@contextmanager
def use_catalog_snapshot(
    repo_root: Path,
    workspace_id: str,
    snapshot: object,
    *,
    profile_id: str = "default",
) -> Iterator[None]:
    """Bind a service-pinned catalog to nested recall and routed-write calls."""

    blocks = catalog_snapshot_blocks(snapshot)
    token = _BOUND_CATALOG.set(
        (
            Path(repo_root).resolve(),
            str(workspace_id),
            str(profile_id),
            tuple(catalog_snapshot_payload(blocks)),
        )
    )
    try:
        yield
    finally:
        _BOUND_CATALOG.reset(token)


def load_memory_blocks(
    repo_root: Path,
    workspace_id: str = "",
    *,
    profile_id: str = "default",
    strict: bool = False,
) -> list[MemoryBlock]:
    bound = _BOUND_CATALOG.get()
    resolved_root = Path(repo_root).resolve()
    using_bound_catalog = (
        bound is not None
        and bound[0] == resolved_root
        and bound[1] == str(workspace_id)
        and bound[2] == str(profile_id)
    )
    by_id: dict[str, MemoryBlock]
    if using_bound_catalog and bound is not None:
        # A service lease pins the policy-owned catalog, not the agent-owned
        # workspace topology.  The latter is deliberately mutable inside one
        # long-running session, so discard its stale lease copy and overlay the
        # canonical SQLite view below.
        by_id = {
            block.block_id: block
            for block in catalog_snapshot_blocks(bound[3])
            if block.block_type not in _WORKSPACE_TOPOLOGY_BLOCK_TYPES
        }
    else:
        sources = [
            default_blocks_path(),
            Path(repo_root) / ".UM-MEM" / "memory" / "blocks.yaml",
        ]
        if workspace_id:
            sources.append(
                Path(repo_root) / "workspaces" / workspace_id / ".memory" / "blocks.yaml"
            )
        by_id = {}
        for source in sources:
            for block in _load_yaml(source, strict=strict):
                if block.block_id in by_id and source != default_blocks_path():
                    current = by_id[block.block_id]
                    # Workspace config may narrow or add blocks, but cannot upgrade passive to directive authority.
                    if (
                        current.authority == "directive_candidate"
                        and block.authority != current.authority
                    ):
                        continue
                by_id[block.block_id] = block
    if workspace_id:
        dynamic_states = load_workspace_topology(
            repo_root,
            workspace_id,
            profile_id=profile_id,
        )
        for state in dynamic_states:
            block = state.block
            if block.block_id in by_id:
                raise ValueError(
                    f"dynamic memory block collides with protected catalog block: {block.block_id}"
                )
            by_id[block.block_id] = block
    blocks = list(by_id.values())
    validate_memory_block_tree(blocks)
    validate_leaf_block_contracts(blocks)
    validate_taxonomy_block_ids(blocks)
    return blocks
