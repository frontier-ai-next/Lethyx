"""Conservative code chunk helpers for external projects."""

import hashlib

from lethyx.memory.external_projects.models import CodeChunkCard

_SECRET_MARKERS = ("secret", "password", "token=", "BEGIN PRIVATE KEY")


def preview_is_safe(text: str) -> bool:
    lower = text.lower()
    return not any(marker.lower() in lower for marker in _SECRET_MARKERS)


def make_code_chunk(
    project_id: str,
    path: str,
    content: str,
    language: str = "",
) -> CodeChunkCard | None:
    preview = content[:1200]
    if not preview_is_safe(preview):
        return None
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    tags = (
        ("github_project", "code_chunk", language) if language else ("github_project", "code_chunk")
    )
    return CodeChunkCard(
        chunk_id=f"chunk_{digest[:16]}",
        project_id=project_id,
        path=path,
        symbol="",
        language=language,
        content_preview=preview,
        content_hash="sha256:" + digest,
        token_count=max(1, len(preview) // 4),
        tags=tags,
    )
