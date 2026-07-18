"""Public, framework-neutral Lethyx API.

The API is intentionally thin: host frameworks own routing and task control,
while Lethyx owns typed memory retention, recall, lifecycle, and graph export.
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lethyx.memory.backends.base import MemoryHit, MemoryQuery
from lethyx.memory.backends.contracts import MemoryBlockRecord, validate_block_record
from lethyx.memory.backends.factory import create_durable_backend


@dataclass(frozen=True)
class RememberResult:
    """Result returned by :meth:`MemoryClient.remember`."""

    saved: bool
    memory_id: str = ""
    backend: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSession:
    """Lightweight session binding used by host frameworks."""

    session_id: str
    actor: str = ""
    project: str = ""
    workspace_id: str = ""


class MemoryClient:
    """Framework-neutral client for Lethyx durable memory.

    Pass an explicit backend in tests or adapters. In normal use, the client
    creates the selected durable backend from ``backend`` or environment.
    """

    def __init__(
        self,
        repo_root: str | Path = ".",
        *,
        workspace_id: str = "default",
        backend: str | None = None,
        backend_instance: Any | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.workspace_id = workspace_id
        self.backend_mode = backend
        self._backend = backend_instance
        self._session: AgentSession | None = None

    @property
    def backend(self) -> Any:
        if self._backend is None:
            self._backend = create_durable_backend(
                self.repo_root,
                workspace_id=self.workspace_id,
                mode=self.backend_mode,
            )
        return self._backend

    @property
    def session(self) -> AgentSession | None:
        return self._session

    def start_session(
        self,
        session_id: str | None = None,
        *,
        actor: str = "",
        project: str = "",
    ) -> AgentSession:
        session = AgentSession(
            session_id=session_id or f"session_{uuid.uuid4().hex[:12]}",
            actor=actor,
            project=project,
            workspace_id=self.workspace_id,
        )
        self._session = session
        return session

    def end_session(self, session_id: str | None = None) -> AgentSession | None:
        current = self._session
        if current is None:
            return None
        if session_id and current.session_id != session_id:
            raise ValueError(f"active session is {current.session_id!r}, not {session_id!r}")
        self._session = None
        return current

    def remember(
        self,
        content: str,
        *,
        title: str = "",
        kind: str = "observation",
        tags: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
        evidence_refs: list[Any] | tuple[Any, ...] | None = None,
        scope: str = "workspace",
        lifecycle: str = "active",
        surface: str = "supplemental_evidence",
        trust_level: str = "agent_claim",
        tier: str = "warm",
        memory_id: str = "",
        run_id: str = "",
        phase_id: str = "",
        subtask_id: str = "",
    ) -> RememberResult:
        """Retain one typed memory block.

        Defaults are useful for agent observations: active for recall, but
        supplemental rather than directive. Directive memory still requires
        explicit verified trust and promotion metadata in the lower contract.
        """

        session = self._session
        meta = dict(metadata or {})
        if session is not None:
            meta.setdefault("session_id", session.session_id)
            if session.actor:
                meta.setdefault("actor", session.actor)
            if session.project:
                meta.setdefault("project", session.project)
        record = MemoryBlockRecord(
            block_id=memory_id,
            title=title or content.strip().splitlines()[0][:96],
            content=content,
            kind=kind,
            lifecycle=lifecycle,
            trust_level=trust_level,
            scope=scope,
            tier=tier,
            surface=surface,
            workspace_id=self.workspace_id,
            run_id=run_id,
            phase_id=phase_id,
            subtask_id=subtask_id,
            source_backend="lethyx",
            tags=tuple(tags or ()),
            evidence_refs=list(evidence_refs or ()),
            metadata=meta,
            verified=trust_level != "agent_claim" and bool(evidence_refs),
        )
        validate_block_record(record)
        raw = dict(self.backend.retain_block(record) or {})
        saved = bool(raw.get("ok") or raw.get("saved") or raw.get("block_id"))
        return RememberResult(
            saved=saved,
            memory_id=str(raw.get("block_id") or record.block_id),
            backend=str(raw.get("backend") or raw.get("source_backend") or self.backend_mode or ""),
            raw=raw,
        )

    def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        tags: list[str] | tuple[str, ...] | None = None,
        run_id: str = "",
        phase_id: str = "",
        max_tokens: int = 2048,
    ) -> list[MemoryHit]:
        """Recall memory as non-directive evidence for a host agent."""

        payload = MemoryQuery(
            query=query,
            workspace_id=self.workspace_id,
            run_id=run_id,
            phase_id=phase_id,
            tags=list(tags or ()),
            max_tokens=max_tokens,
        )
        if hasattr(self.backend, "recall_evidence"):
            hits = self.backend.recall_evidence(payload)
        else:
            hits = self.backend.recall(payload)
        return list(hits)[: max(0, limit)]

    def forget(self, memory_id: str, *, reason: str = "user_requested") -> dict[str, Any]:
        """Retract a memory block without deleting its audit trail."""

        return dict(
            self.backend.update_lifecycle(
                memory_id,
                "retracted",
                reason=reason,
                surface="archive_hint",
            )
            or {}
        )

    def promote(
        self,
        memory_id: str,
        *,
        reason: str,
        evidence_refs: list[Any] | tuple[Any, ...],
        surface: str = "supplemental_evidence",
    ) -> dict[str, Any]:
        """Promote a memory block through an explicit evidence-bearing action."""

        if not evidence_refs:
            raise ValueError("promotion requires evidence_refs")
        return dict(
            self.backend.update_lifecycle(
                memory_id,
                "active",
                reason=reason,
                evidence_refs=list(evidence_refs),
                surface=surface,
            )
            or {}
        )

    def build_context(
        self,
        query: str,
        *,
        limit: int = 8,
        max_tokens: int = 2048,
    ) -> list[dict[str, Any]]:
        """Build a prompt-safe context bundle from recalled evidence."""

        items: list[dict[str, Any]] = []
        for hit in self.recall(query, limit=limit, max_tokens=max_tokens):
            items.append(
                {
                    "text": hit.text,
                    "source": hit.source,
                    "score": hit.score,
                    "kind": hit.kind,
                    "tags": list(hit.tags),
                    "metadata": dict(hit.metadata),
                    "surface": "supplemental_evidence",
                }
            )
        return items

    def export_graph(self) -> dict[str, Any]:
        """Export backend blocks and available graph edges."""

        blocks = list(self.backend.export_blocks() or [])
        edges: list[Any] = []
        for block in blocks:
            block_id = getattr(block, "block_id", "") or (
                block.get("block_id", "") if isinstance(block, dict) else ""
            )
            if block_id and hasattr(self.backend, "edges_for"):
                edges.extend(self.backend.edges_for(str(block_id)) or [])
        return {"blocks": blocks, "edges": edges}


def remember(content: str, **kwargs: Any) -> RememberResult:
    """Retain memory through a one-off default client."""

    client_kwargs = _client_kwargs(kwargs)
    return MemoryClient(**client_kwargs).remember(content, **kwargs)


def recall(query: str, **kwargs: Any) -> list[MemoryHit]:
    """Recall memory through a one-off default client."""

    client_kwargs = _client_kwargs(kwargs)
    return MemoryClient(**client_kwargs).recall(query, **kwargs)


def _client_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    client_keys = {"repo_root", "workspace_id", "backend", "backend_instance"}
    client_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in client_keys}
    return client_kwargs
