"""Models for passive GitHub/external project memory."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalProjectRef:
    provider: str
    owner: str
    repo: str
    ref: str = ""
    url: str = ""
    license: str = ""
    trust_level: str = "public_source"
    indexed_at: str = ""


@dataclass(frozen=True)
class RepoCapsule:
    project_id: str
    title: str
    summary: str
    languages: tuple[str, ...]
    frameworks: tuple[str, ...]
    entrypoints: tuple[str, ...]
    key_files: tuple[str, ...]
    reusable_patterns: tuple[str, ...]
    risks: tuple[str, ...]
    source_url: str
    commit_sha: str
    license: str


@dataclass(frozen=True)
class CodeChunkCard:
    chunk_id: str
    project_id: str
    path: str
    symbol: str
    language: str
    content_preview: str
    content_hash: str
    token_count: int
    tags: tuple[str, ...]
