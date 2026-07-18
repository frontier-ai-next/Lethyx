"""Write helpers for MemPalace-backed memory records."""

from pathlib import Path
from typing import Any

from lethyx.memory.palace.facade import MemPalace


def canonical_palace_add(
    repo_root: Path,
    *,
    workspace_id: str,
    content: str,
    title: str = "",
    kind: str = "observation",
    store: str = "palace.idea",
    tags: list[str] | tuple[str, ...] | str | None = None,
    phase: str = "",
    source_path: str = "",
    extra: dict[str, Any] | None = None,
    run_id: str = "",
    verified: bool = False,
    node_id: str = "",
) -> dict[str, Any]:
    """Add a canonical node to MemPalace and return a stable result payload."""

    tag_list = _normalize_tags(tags)
    requested_id = str(node_id or (extra or {}).get("canonical_id") or "").strip()
    metadata = {
        "title": str(title or "").strip(),
        "kind": str(kind or "observation"),
        "type": str(kind or "observation"),
        **(extra or {}),
    }
    palace = MemPalace(Path(repo_root), workspace_id)
    try:
        if requested_id and palace.get(requested_id, stores=[store]):
            return {
                "saved": True,
                "node_id": requested_id,
                "canonical_id": requested_id,
                "store": store,
                "workspace_id": workspace_id,
                "deduped": True,
            }
        node_id = palace.add(
            store=store,
            content=str(content or ""),
            tags=tag_list,
            phase=phase or None,
            source_path=source_path or None,
            run_id=run_id or None,
            verified=verified,
            extra=metadata,
            kind=kind,
            node_id=requested_id or None,
        )
    finally:
        palace.close()
    return {
        "saved": True,
        "node_id": node_id,
        "canonical_id": node_id,
        "store": store,
        "workspace_id": workspace_id,
    }


def _normalize_tags(tags: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        return [part.strip() for part in tags.split(",") if part.strip()]
    return [str(part).strip() for part in tags if str(part).strip()]
