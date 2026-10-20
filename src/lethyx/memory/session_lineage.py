"""Exact parent/child session continuity without foreign-session scans."""

from dataclasses import dataclass

from lethyx.contracts import EvidenceRef


def _required(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


@dataclass(frozen=True)
class ParentChildSessionLink:
    link_id: str
    workspace_id: str
    parent_session_id: str
    child_session_id: str
    evidence_refs: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "link_id",
            "workspace_id",
            "parent_session_id",
            "child_session_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.parent_session_id == self.child_session_id:
            raise ValueError("session lineage requires distinct parent and child")
        if isinstance(self.evidence_refs, (str, bytes)) or not isinstance(
            self.evidence_refs, tuple
        ):
            raise ValueError("session lineage evidence_refs must be a tuple")
        if not self.evidence_refs or any(
            not isinstance(ref, EvidenceRef) for ref in self.evidence_refs
        ):
            raise ValueError("session lineage requires EvidenceRef values")
        object.__setattr__(self, "evidence_refs", tuple(dict.fromkeys(self.evidence_refs)))


@dataclass(frozen=True)
class SessionContinuityResult:
    workspace_id: str
    session_ids: tuple[str, ...]
    consulted_link_ids: tuple[str, ...]
    foreign_session_scan: bool = False


class SessionContinuityResolver:
    def resolve(
        self,
        *,
        workspace_id: str,
        current_session_id: str,
        links: tuple[ParentChildSessionLink, ...],
        max_depth: int = 16,
    ) -> SessionContinuityResult:
        workspace = _required(workspace_id, field_name="workspace_id")
        current = _required(current_session_id, field_name="current_session_id")
        if (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or not 1 <= max_depth <= 64
        ):
            raise ValueError("max_depth must be between one and 64")
        if not isinstance(links, tuple) or any(
            not isinstance(link, ParentChildSessionLink) for link in links
        ):
            raise ValueError("links must contain ParentChildSessionLink values")
        parent_by_child: dict[str, ParentChildSessionLink] = {}
        for link in links:
            if link.workspace_id != workspace:
                continue
            if link.child_session_id in parent_by_child:
                raise ValueError("session may have only one parent in a workspace")
            parent_by_child[link.child_session_id] = link
        sessions = [current]
        consulted: list[str] = []
        while len(sessions) < max_depth:
            parent_link = parent_by_child.get(sessions[-1])
            if parent_link is None:
                break
            if parent_link.parent_session_id in sessions:
                raise ValueError("session lineage cycle detected")
            sessions.append(parent_link.parent_session_id)
            consulted.append(parent_link.link_id)
        return SessionContinuityResult(
            workspace_id=workspace,
            session_ids=tuple(sessions),
            consulted_link_ids=tuple(consulted),
        )
