"""Map external project artifacts to passive memory rows."""

from dataclasses import asdict
from typing import Any

from lethyx.memory.external_projects.models import CodeChunkCard, RepoCapsule


def capsule_to_memory_row(capsule: RepoCapsule) -> dict[str, Any]:
    return {
        "store": "palace.codeptr",
        "tier": "warm",
        "surface": "supplemental_evidence",
        "kind": "repo_capsule",
        "tags": ["github_project", "repo_capsule", *capsule.languages],
        "metadata": asdict(capsule),
    }


def chunk_to_memory_row(chunk: CodeChunkCard) -> dict[str, Any]:
    return {
        "store": "palace.codeptr",
        "tier": "warm",
        "surface": "supplemental_evidence",
        "kind": "code_chunk",
        "tags": list(chunk.tags),
        "metadata": asdict(chunk),
    }
