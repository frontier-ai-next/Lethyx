"""Repo capsule builders."""

from lethyx.memory.external_projects.models import ExternalProjectRef, RepoCapsule


def build_repo_capsule(
    ref: ExternalProjectRef,
    *,
    summary: str,
    languages: tuple[str, ...] = (),
    key_files: tuple[str, ...] = (),
) -> RepoCapsule:
    project_id = f"{ref.provider}:{ref.owner}/{ref.repo}:{ref.ref or 'HEAD'}"
    return RepoCapsule(
        project_id,
        f"{ref.owner}/{ref.repo}",
        summary,
        languages,
        (),
        (),
        key_files,
        (),
        (),
        ref.url,
        ref.ref,
        ref.license,
    )
