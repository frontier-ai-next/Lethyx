"""Language-neutral semantic retrieval for the local record window.

The lexical tier in :mod:`um_mem.memory.text` scores character n-grams, so a
query only matches a record when the two share characters. A Russian question
("где находится конфигурация") shares nothing with the English record "OpenCode
config locations", and the answer was silently dropped even though the record
was sitting right there. That is a script problem, not a corpus problem.

This module adds one more local tier: multilingual sentence embeddings
computed on the machine, cached per workspace, and scored by cosine
similarity. The model is multilingual by construction, so the tier behaves the
same for Latin, Cyrillic, Greek, Arabic, Hebrew, Devanagari, Han, Kana and
Hangul queries. There is no stopword list, no per-language branch and no
alphabet range anywhere in this file.

The embeddings are advisory evidence, exactly like the lexical tier: they
propose candidates, and the stored record text remains the only thing an
answer may cite.

The model is optional. When ``fastembed`` is not installed, or the operator
disables the tier, every helper here degrades to "no semantic evidence" and
the lexical tier keeps working unchanged.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sqlite3
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from um_mem.memory.paths import workspace_memory_root

#: Default multilingual model. E5 models are trained for retrieval and keep a
#: usable similarity gap between a matching and a non-matching passage in every
#: supported language.
DEFAULT_MODEL = "intfloat/multilingual-e5-large"

#: Per-model relevance floors. Cosine similarity is not comparable across
#: models, so the floor belongs to the model, never to a language.
MODEL_FLOORS: dict[str, float] = {
    "intfloat/multilingual-e5-large": 0.72,
    "intfloat/multilingual-e5-small": 0.60,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 0.30,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 0.50,
    "minishlab/potion-multilingual-128M": 0.45,
}

#: Floor used when the model is unknown to this table.
DEFAULT_FLOOR = 0.65

#: Extra margin applied when the question is written in the same script as the
#: stored text. The lexical tier already owns those queries, so a semantic hit
#: that only the model found has to be clearly relevant to survive. A question
#: in another script has no lexical tier to fall back on and keeps the base
#: floor.
LATIN_SCRIPT_MARGIN = 0.05

#: How much of the semantic score comes from the whole record and how much
#: from the human-written title. The title is a short, deliberate label, so it
#: carries a lot of meaning that a long body dilutes.
TITLE_BLEND = 0.6

#: E5 models expect an asymmetric prefix for retrieval.
_QUERY_PREFIX = "query: "
_DOCUMENT_PREFIX = "passage: "

_EMBEDDERS: dict[tuple[str, str], Embedder] = {}
_EMBEDDER_LOCK = threading.Lock()


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off", "disabled"}


def semantic_enabled() -> bool:
    """Return whether the operator allows the local semantic tier."""

    return _env_flag("UM_MEM_SEMANTIC", True)


def semantic_warmup_enabled() -> bool:
    """Return whether the model may be loaded in the background at start.

    Loading the model in the background makes the first question fast, but it
    also downloads the model when it is not cached yet. An operator who wants
    to control when that happens prefetches it with ``um-memory semantic-model``
    or switches the warmup off.
    """

    return _env_flag("UM_MEM_SEMANTIC_WARMUP", True)


def idle_release_seconds() -> int:
    """Return how long an unused model may stay loaded.

    The model is a few gigabytes of resident memory. It is only useful while
    a question is being scored, so an unused model is released and loaded
    again on the next question.
    """

    raw = os.getenv("UM_MEM_SEMANTIC_IDLE_SECONDS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return 120


def default_model() -> str:
    """Return the configured embedding model name."""

    return os.getenv("UM_MEM_SEMANTIC_MODEL", "").strip() or DEFAULT_MODEL


def default_cache_dir(repo_root: Path | str | None = None) -> Path:
    """Return the directory that holds the downloaded model files.

    The model files belong next to the store, not in a random per-user cache,
    so an operator keeps one copy of a multi-gigabyte model with the rest of
    the local memory installation.
    """

    raw = (
        os.getenv("UM_MEM_SEMANTIC_CACHE_DIR", "").strip()
        or os.getenv("MNEMOSYNE_FASTEMBED_CACHE_DIR", "").strip()
    )
    if raw:
        return Path(raw).expanduser()
    if repo_root:
        try:
            return Path(repo_root).resolve().parent / "cache" / "fastembed"
        except OSError:  # pragma: no cover - platform dependent
            pass
    return Path.home() / ".cache" / "fastembed"


def relevance_floor(model: str | None = None) -> float:
    """Return the minimum cosine similarity that counts as semantic evidence."""

    raw = os.getenv("UM_MEM_SEMANTIC_FLOOR", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    selected = (model or default_model()).strip()
    return MODEL_FLOORS.get(selected, DEFAULT_FLOOR)


def query_uses_non_ascii_script(query: object) -> bool:
    """Return whether the query carries letters outside the ASCII range.

    A query written in any other script cannot be answered by matching ASCII
    characters, so the lexical tier needs help. The check looks at Unicode
    letter properties, not at a list of languages.
    """

    return any(char.isalpha() and not char.isascii() for char in str(query or ""))


#: Probes that must not match anything. They are deliberately in several
#: scripts: a floor that only holds for one language is not a floor.
IRRELEVANT_PROBES = (
    "what is the capital of France",
    "how do I bake sourdough bread",
    "погода в Москве на завтра",
    "quelle est la capitale de la France",
    "如何烹饪意大利面",
)


def calibration_path(repo_root: Path | str, workspace_id: str) -> Path:
    """Return the file that holds the measured relevance floor."""

    return semantic_index_path(repo_root, workspace_id).parent / "calibration.json"


def load_calibration(repo_root: Path | str, workspace_id: str) -> dict[str, Any]:
    """Return the measured floor of this workspace, if it was calibrated."""

    path = calibration_path(repo_root, workspace_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def store_calibration(repo_root: Path | str, workspace_id: str, payload: Mapping[str, Any]) -> Path:
    """Write the measured floor of this workspace."""

    path = calibration_path(repo_root, workspace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def calibrated_floor(
    repo_root: Path | str, workspace_id: str, model: str | None = None
) -> float | None:
    """Return the measured floor of this workspace, if there is one."""

    payload = load_calibration(repo_root, workspace_id)
    if payload.get("model") and model and payload["model"] != model:
        return None
    value = payload.get("floor")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _best_similarity(
    query_vector: Sequence[float],
    vectors: Mapping[str, Mapping[str, tuple[str, list[float]]]],
) -> float:
    """Return the best blended similarity of one query against the index."""

    best = 0.0
    for variants in vectors.values():
        full = variants.get("full")
        if not full:
            continue
        score = TITLE_BLEND * _cosine(query_vector, full[1])
        title = variants.get("title")
        if title:
            score += (1.0 - TITLE_BLEND) * _cosine(query_vector, title[1])
        best = max(best, score)
    return best


def calibrate(
    records: Sequence[Mapping[str, Any]],
    index: SemanticIndex,
    embedder: Embedder,
    *,
    model: str | None = None,
    profile: str = "default",
) -> dict[str, Any]:
    """Measure the relevance floor for one workspace.

    The relevant probes are the titles of the stored records: a record always
    matches itself. The irrelevant probes are questions about something else
    entirely, in several scripts. The floor is the gap between the two
    groups, so it is measured for this model and this corpus instead of being
    a fixed number.
    """

    selected_model = (model or embedder.model).strip()
    vectors = index.all_vectors(selected_model, profile)
    if not vectors:
        return {"ok": False, "error": "the semantic index is empty"}
    titles = [title for title in (record_title(record) for record in records) if title]
    titles = list(dict.fromkeys(titles))[:32]
    if not titles:
        return {"ok": False, "error": "the workspace has no titled records"}
    probes = titles + list(IRRELEVANT_PROBES)
    query_vectors = embedder.embed(probes, kind="query")
    if not query_vectors:
        return {"ok": False, "error": "the model returned no vectors"}
    relevant: list[float] = []
    irrelevant: list[float] = []
    for position, probe in enumerate(probes):
        score = _best_similarity(query_vectors[position], vectors)
        if position < len(titles):
            relevant.append(score)
        else:
            irrelevant.append(score)
    if not relevant or not irrelevant:
        return {"ok": False, "error": "not enough probes"}
    background = _median(irrelevant)
    # The relevant probes are titles, so they match almost perfectly and say
    # little about a real question. The floor therefore sits just above the
    # measured background of unrelated questions, never below the model
    # default, and the strict midpoint is reported for the operator.
    floor = max(relevance_floor(selected_model), background + 0.02)
    strict = 0.5 * (max(irrelevant) + min(relevant))
    return {
        "ok": True,
        "model": selected_model,
        "profile": profile,
        "floor": round(float(floor), 4),
        "strict_floor": round(float(strict), 4),
        "relevant": {
            "count": len(relevant),
            "min": round(min(relevant), 4),
            "median": round(_median(relevant), 4),
        },
        "irrelevant": {
            "count": len(irrelevant),
            "max": round(max(irrelevant), 4),
            "median": round(background, 4),
        },
        "default_floor": relevance_floor(selected_model),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def effective_floor(
    model: str | None = None, *, cross_script: bool, calibrated: float | None = None
) -> float:
    """Return the relevance floor for one query.

    Same-script questions keep the model floor, because the lexical tier
    already answers them. Questions in another script get the base floor,
    because nothing else can reach across scripts. A measured floor for this
    workspace always wins over the model default.
    """

    floor = relevance_floor(model)
    if calibrated is not None:
        floor = max(floor, float(calibrated))
    if not cross_script:
        floor += LATIN_SCRIPT_MARGIN
    return floor


def _normalize(vector: Sequence[float]) -> list[float]:
    total = sum(value * value for value in vector) ** 0.5
    if not total:
        return [0.0 for _ in vector]
    return [float(value) / total for value in vector]


class Embedder:
    """Lazily loaded multilingual sentence embedder."""

    def __init__(self, model: str | None = None, cache_dir: Path | None = None) -> None:
        self.model = (model or default_model()).strip() or DEFAULT_MODEL
        self.cache_dir = Path(cache_dir or default_cache_dir())
        self._model: Any | None = None
        self._lock = threading.RLock()
        self._failed = False
        self._load_seconds = 0.0
        self._last_used_at = 0.0

    @property
    def load_seconds(self) -> float:
        """Return how long the model took to load, for diagnostics."""

        return self._load_seconds

    @property
    def idle_seconds(self) -> float:
        """Return how long the loaded model has been unused."""

        if self._model is None:
            return 0.0
        return time.monotonic() - self._last_used_at

    def release_if_idle(self, idle_seconds: float) -> bool:
        """Drop the loaded model when it has been unused for a while.

        The model costs a few gigabytes of resident memory and is only useful
        while a question is being scored. Releasing it keeps a long-lived
        service light; the next question loads it again.
        """

        if self._model is None or idle_seconds <= 0:
            return False
        with self._lock:
            if self._model is None:
                return False
            if time.monotonic() - self._last_used_at < idle_seconds:
                return False
            self._model = None
            self._load_seconds = 0.0
        gc.collect()
        return True

    def available(self) -> bool:
        """Return whether the embedding runtime is importable."""

        if self._failed:
            return False
        import importlib.util

        return importlib.util.find_spec("fastembed") is not None

    def _prefixed(self, text: str, kind: str) -> str:
        if "e5" not in self.model.lower():
            return text
        return _QUERY_PREFIX + text if kind == "query" else _DOCUMENT_PREFIX + text

    def _ensure_model(self) -> Any | None:
        if self._model is not None or self._failed:
            return self._model
        with self._lock:
            if self._model is not None or self._failed:
                return self._model
            try:
                from fastembed import TextEmbedding  # type: ignore[import-not-found]

                started = time.perf_counter()
                self._model = TextEmbedding(model_name=self.model, cache_dir=str(self.cache_dir))
                self._load_seconds = time.perf_counter() - started
            except Exception:  # pragma: no cover - optional dependency
                self._failed = True
                self._model = None
        return self._model

    def embed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        """Return one normalized embedding per input text.

        An empty list is returned when the model is unavailable so callers can
        degrade to the lexical tier instead of failing the request.
        """

        prepared = [self._prefixed(str(text or ""), kind) for text in texts]
        if not prepared:
            return []
        with self._lock:
            model = self._ensure_model()
            if model is None:
                return []
            vectors: list[list[float]] = []
            for vector in model.embed(prepared):
                values = [float(value) for value in vector]
                vectors.append(_normalize(values))
            self._last_used_at = time.monotonic()
        return vectors


def get_embedder(model: str | None = None, cache_dir: Path | str | None = None) -> Embedder | None:
    """Return the process-wide embedder for one model."""

    if not semantic_enabled():
        return None
    selected = (model or default_model()).strip() or DEFAULT_MODEL
    resolved = Path(cache_dir) if cache_dir else default_cache_dir()
    key = (selected, str(resolved))
    with _EMBEDDER_LOCK:
        embedder = _EMBEDDERS.get(key)
        if embedder is None:
            embedder = Embedder(selected, resolved)
            _EMBEDDERS[key] = embedder
    return embedder if embedder.available() else None


def semantic_ready(repo_root: Path | str | None = None) -> bool:
    """Return whether the semantic tier can run for this installation."""

    if not semantic_enabled():
        return False
    return get_embedder(cache_dir=default_cache_dir(repo_root)) is not None


def known_embedders() -> list[Embedder]:
    """Return every embedder this process has created."""

    with _EMBEDDER_LOCK:
        return list(_EMBEDDERS.values())


def semantic_index_path(repo_root: Path | str, workspace_id: str) -> Path:
    """Return the sidecar index that caches record embeddings."""

    root = Path(repo_root)
    workspace = str(workspace_id or "default").strip() or "default"
    return workspace_memory_root(root, workspace) / "semantic" / "embeddings.sqlite"


def content_digest(text: object) -> str:
    """Return the digest that decides whether an embedding is still valid."""

    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


class SemanticIndex:
    """SQLite sidecar that caches record embeddings by content digest."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(semantic_vectors)")}
        if columns and (
            "variant" not in columns or "profile" not in columns or "kind" not in columns
        ):
            # The index is a cache. An older layout is dropped instead of
            # migrated, because every stored vector is cheap to recompute.
            connection.execute("DROP TABLE semantic_vectors")
            connection.execute("DROP TABLE IF EXISTS semantic_versions")
            connection.commit()
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_vectors (
                model TEXT NOT NULL,
                profile TEXT NOT NULL,
                record_id TEXT NOT NULL,
                variant TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT '',
                content_sha256 TEXT NOT NULL,
                vector BLOB NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (model, profile, record_id, variant)
            )
            """
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS semantic_versions ("
            "model TEXT NOT NULL, profile TEXT NOT NULL, revision TEXT NOT NULL, "
            "PRIMARY KEY (model, profile))"
        )
        return connection

    def load(self, model: str, profile: str, items: Mapping[str, str]) -> dict[str, list[float]]:
        """Return cached vectors for the given keys and content digests."""

        if not items:
            return {}
        identifiers: list[str] = []
        for key in items:
            record_id, _, _variant = str(key).partition("|")
            if record_id and record_id not in identifiers:
                identifiers.append(record_id)
        with self._lock, self._connect() as connection:
            found: dict[str, list[float]] = {}
            for start in range(0, len(identifiers), 400):
                chunk = identifiers[start : start + 400]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    "SELECT record_id, variant, content_sha256, vector FROM semantic_vectors "
                    f"WHERE model = ? AND profile = ? AND record_id IN ({placeholders})",
                    (model, profile, *chunk),
                ).fetchall()
                for record_id, variant, digest, blob in rows:
                    key = f"{record_id}|{variant}"
                    if str(digest) != items.get(key, ""):
                        continue
                    found[key] = list(struct.unpack(f"<{len(blob) // 4}f", blob))
            return found

    def store(
        self,
        model: str,
        profile: str,
        entries: Mapping[str, tuple[str, Sequence[float]]],
        kinds: Mapping[str, str] | None = None,
    ) -> None:
        """Persist embeddings together with the digest they were computed from."""

        if not entries:
            return
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rows = []
        for key, (digest, vector) in entries.items():
            record_id, _, variant = str(key).partition("|")
            payload = struct.pack(f"<{len(vector)}f", *[float(value) for value in vector])
            kind = str((kinds or {}).get(key, ""))
            rows.append(
                (
                    model,
                    profile,
                    record_id,
                    variant or "full",
                    kind,
                    str(digest),
                    payload,
                    now,
                )
            )
        with self._lock, self._connect() as connection:
            connection.executemany(
                "INSERT INTO semantic_vectors "
                "(model, profile, record_id, variant, kind, content_sha256, vector, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(model, profile, record_id, variant) DO UPDATE SET "
                "kind = excluded.kind, "
                "content_sha256 = excluded.content_sha256, "
                "vector = excluded.vector, updated_at = excluded.updated_at",
                rows,
            )
            self._change_version(connection, model, profile)
            connection.commit()

    def all_vectors(
        self, model: str, profile: str
    ) -> dict[str, dict[str, tuple[str, list[float]]]]:
        """Return every indexed vector of one model and profile.

        The result maps a record id to its ``full`` and ``title`` variants.
        """

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT record_id, variant, content_sha256, vector FROM semantic_vectors "
                "WHERE model = ? AND profile = ?",
                (model, profile),
            ).fetchall()
        found: dict[str, dict[str, tuple[str, list[float]]]] = {}
        for record_id, variant, digest, blob in rows:
            values = list(struct.unpack(f"<{len(blob) // 4}f", blob))
            found.setdefault(str(record_id), {})[str(variant)] = (str(digest), values)
        return found

    def kinds(self, model: str, profile: str) -> dict[str, str]:
        """Return the record kind of every indexed record."""

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT record_id, kind FROM semantic_vectors WHERE model = ? AND profile = ?",
                (model, profile),
            ).fetchall()
        return {str(record_id): str(kind) for record_id, kind in rows}

    def delete(self, model: str, profile: str, record_id: str) -> None:
        """Forget the vectors of one record."""

        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM semantic_vectors WHERE model = ? AND profile = ? AND record_id = ?",
                (model, profile, str(record_id)),
            )
            self._change_version(connection, model, profile)
            connection.commit()

    @staticmethod
    def _change_version(connection: sqlite3.Connection, model: str, profile: str) -> None:
        # A durable token in the same transaction catches updates from other
        # processes, including multiple replacements within one clock tick.
        connection.execute(
            "INSERT INTO semantic_versions (model, profile, revision) "
            "VALUES (?, ?, lower(hex(randomblob(16)))) "
            "ON CONFLICT(model, profile) DO UPDATE SET revision = excluded.revision",
            (model, profile),
        )

    def count(self, model: str, profile: str) -> int:
        """Return how many records are indexed for one model and profile."""

        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(DISTINCT record_id) FROM semantic_vectors "
                "WHERE model = ? AND profile = ?",
                (model, profile),
            ).fetchone()
        return int(row[0]) if row else 0

    def version(self, model: str, profile: str) -> str:
        """Return a token that changes whenever the index changes."""

        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT revision FROM semantic_versions WHERE model = ? AND profile = ?",
                (model, profile),
            ).fetchone()
        return str(row[0]) if row else "unversioned"


#: Process-wide cache of the parsed index. Reading and unpacking every vector
#: on every question is the one cost that does not belong in a query path.
_VECTOR_CACHE: dict[tuple[str, str, str], tuple[str, dict[str, Any]]] = {}
_VECTOR_CACHE_LOCK = threading.Lock()


def cached_vectors(
    index: SemanticIndex, model: str, profile: str
) -> dict[str, dict[str, tuple[str, list[float]]]]:
    """Return the indexed vectors, reusing the parsed copy when possible."""

    version = index.version(model, profile)
    key = (str(index.path.resolve()), model, profile)
    with _VECTOR_CACHE_LOCK:
        cached = _VECTOR_CACHE.get(key)
        if cached is not None and cached[0] == version:
            return cached[1]
    vectors = index.all_vectors(model, profile)
    with _VECTOR_CACHE_LOCK:
        _VECTOR_CACHE[key] = (version, vectors)
    return vectors


def score_vectors(
    query_vector: Sequence[float],
    vectors: Mapping[str, Mapping[str, tuple[str, list[float]]]],
) -> list[tuple[str, float]]:
    """Return every record with its blended similarity to the query."""

    if not vectors:
        return []
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - optional accelerator
        scored: list[tuple[str, float]] = []
        for record_id, variants in vectors.items():
            full = variants.get("full")
            if not full:
                continue
            score = TITLE_BLEND * _cosine(query_vector, full[1])
            title = variants.get("title")
            if title:
                score += (1.0 - TITLE_BLEND) * _cosine(query_vector, title[1])
            scored.append((record_id, score))
        return scored
    identifiers = []
    full_rows = []
    title_rows = []
    for record_id, variants in vectors.items():
        full = variants.get("full")
        if not full:
            continue
        identifiers.append(record_id)
        full_rows.append(full[1])
        title_variant = variants.get("title")
        title_rows.append(title_variant[1] if title_variant else None)
    if not identifiers:
        return []
    matrix = np.asarray(full_rows, dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    scores = matrix @ query
    blended = scores * TITLE_BLEND
    has_title = [index for index, row in enumerate(title_rows) if row is not None]
    if has_title:
        title_matrix = np.asarray([title_rows[index] for index in has_title], dtype=np.float32)
        blended[has_title] += (1.0 - TITLE_BLEND) * (title_matrix @ query)
    return [
        (identifiers[position], float(blended[position])) for position in range(len(identifiers))
    ]


def record_identity(record: Mapping[str, Any]) -> str:
    """Return the stable identifier of one record."""

    for key in ("record_id", "block_id", "id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def record_text(record: Mapping[str, Any]) -> str:
    """Return the text that represents one record for embedding."""

    title = str(record.get("title") or "").strip()
    content = str(record.get("content") or record.get("text") or "").strip()
    return f"{title}\n{content}".strip()


def record_title(record: Mapping[str, Any]) -> str:
    """Return the human-written title of one record."""

    return str(record.get("title") or "").strip()


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(value * value for value in left) ** 0.5
    norm_right = sum(value * value for value in right) ** 0.5
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def rank_semantic(
    query: str,
    records: Sequence[Mapping[str, Any]],
    *,
    embedder: Embedder,
    index: SemanticIndex,
    model: str | None = None,
    profile: str = "default",
    floor: float | None = None,
    limit: int = 8,
) -> list[tuple[str, float]]:
    """Rank records by embedding similarity to the query.

    The score blends the similarity to the whole record with the similarity to
    its title, because a short human-written title carries more intent than a
    long body. Returns ``(record_id, score)`` pairs for the records that clear
    the relevance floor, best first.
    """

    if not records or not str(query or "").strip():
        return []
    selected_model = (model or embedder.model).strip()
    threshold = relevance_floor(selected_model) if floor is None else float(floor)

    identifiers: list[str] = []
    texts: dict[str, str] = {}
    digests: dict[str, str] = {}
    for record in records:
        identifier = record_identity(record)
        full = record_text(record)
        title = record_title(record)
        if not identifier or not full:
            continue
        identifiers.append(identifier)
        texts[f"{identifier}|full"] = full
        digests[f"{identifier}|full"] = content_digest(full)
        if title and title != full:
            texts[f"{identifier}|title"] = title
            digests[f"{identifier}|title"] = content_digest(title)
    if not identifiers:
        return []

    cached = index.load(selected_model, profile, digests)
    missing = [key for key in digests if key not in cached]
    if missing:
        vectors = embedder.embed([texts[key] for key in missing])
        entries = {
            key: (digests[key], vectors[position])
            for position, key in enumerate(missing)
            if position < len(vectors)
        }
        if entries:
            index.store(selected_model, profile, entries)
            cached.update({key: vector for key, (_, vector) in entries.items()})

    query_vectors = embedder.embed([str(query)], kind="query")
    if not query_vectors:
        return []
    query_vector = query_vectors[0]

    scored: list[tuple[str, float]] = []
    for identifier in identifiers:
        full_vector = cached.get(f"{identifier}|full")
        if not full_vector:
            continue
        title_vector = cached.get(f"{identifier}|title")
        score = TITLE_BLEND * _cosine(query_vector, full_vector)
        if title_vector:
            score += (1.0 - TITLE_BLEND) * _cosine(query_vector, title_vector)
        if score >= threshold:
            scored.append((identifier, float(score)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(1, int(limit))]


def embed_record(
    index: SemanticIndex,
    embedder: Embedder,
    *,
    model: str | None = None,
    profile: str = "default",
    record_id: str,
    title: str,
    content: str,
    kind: str = "",
) -> bool:
    """Embed one record and store it in the sidecar index.

    This is what keeps the semantic index complete: every write adds its
    vector, so a cross-language question reaches old records and not only the
    newest window.
    """

    identifier = str(record_id or "").strip()
    full = f"{str(title or '').strip()}\n{str(content or '').strip()}".strip()
    if not identifier or not full:
        return False
    selected_model = (model or embedder.model).strip()
    texts = {f"{identifier}|full": full}
    if str(title or "").strip() and str(title or "").strip() != full:
        texts[f"{identifier}|title"] = str(title).strip()
    digests = {key: content_digest(value) for key, value in texts.items()}
    vectors = embedder.embed([texts[key] for key in texts])
    entries = {
        key: (digests[key], vectors[position])
        for position, key in enumerate(texts)
        if position < len(vectors)
    }
    if not entries:
        return False
    index.store(
        selected_model,
        profile,
        entries,
        kinds={key: str(kind or "") for key in entries},
    )
    return True


def rank_global(
    query: str,
    *,
    embedder: Embedder,
    index: SemanticIndex,
    model: str | None = None,
    profile: str = "default",
    floor: float | None = None,
    limit: int = 8,
    loader: Callable[[Sequence[str]], Sequence[Mapping[str, Any]]],
    kinds: Sequence[str] | None = None,
) -> list[tuple[str, float, Mapping[str, Any]]]:
    """Score every indexed record and return the best live ones.

    The index covers the whole workspace, so a question in any script reaches
    records that were written long ago. The stored text is always read back
    from the canonical store, and a record whose text changed since it was
    embedded is refreshed on the way. ``kinds`` narrows the search to one
    kind of record, for example the stored procedures.
    """

    if not str(query or "").strip():
        return []
    selected_model = (model or embedder.model).strip()
    threshold = relevance_floor(selected_model) if floor is None else float(floor)
    vectors = cached_vectors(index, selected_model, profile)
    if not vectors:
        return []
    wanted_kinds = {str(value).strip() for value in (kinds or ()) if str(value).strip()}
    record_kinds = index.kinds(selected_model, profile) if wanted_kinds else {}
    if wanted_kinds:
        vectors = {
            record_id: variants
            for record_id, variants in vectors.items()
            if record_kinds.get(record_id, "") in wanted_kinds
        }
        if not vectors:
            return []
    query_vectors = embedder.embed([str(query)], kind="query")
    if not query_vectors:
        return []
    query_vector = query_vectors[0]
    scored = [
        (record_id, score)
        for record_id, score in score_vectors(query_vector, vectors)
        if score >= threshold
    ]
    if not scored:
        return []
    scored.sort(key=lambda item: item[1], reverse=True)
    candidates = scored[: max(limit * 3, limit)]
    try:
        loaded = list(loader([record_id for record_id, _ in candidates]))
    except Exception:  # pragma: no cover - defensive boundary
        return []
    by_identity = {record_identity(record): record for record in loaded}
    stale: list[Mapping[str, Any]] = []
    for record_id, _score in candidates:
        record = by_identity.get(record_id)
        if record is None:
            continue
        stored = vectors.get(record_id, {}).get("full", ("",))[0]
        if content_digest(record_text(record)) != stored:
            stale.append(record)
    if stale:
        # A record that changed since it was embedded is re-embedded now, so
        # the index repairs itself instead of drifting.
        for record in stale:
            embed_record(
                index,
                embedder,
                model=selected_model,
                profile=profile,
                record_id=record_identity(record),
                title=record_title(record),
                content=str(record.get("content") or record.get("text") or ""),
                kind=str(record.get("kind") or ""),
            )
    return [
        (record_id, score, by_identity[record_id])
        for record_id, score in candidates
        if record_id in by_identity
    ][: max(1, int(limit))]


def suggest_blocks(
    records: Sequence[Mapping[str, Any]],
    semantic_scores: Mapping[str, float],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Group semantic hits by their logical block.

    The block that carries the best matching records is the one worth opening
    next, which is what a human would do with a labelled memory tree.
    """

    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = record_identity(record)
        score = semantic_scores.get(identifier)
        if score is None:
            continue
        membership = record.get("block_membership")
        block_id = ""
        if isinstance(membership, Mapping):
            block_id = str(membership.get("primary_block_id") or "").strip()
        if not block_id:
            block_id = str(record.get("logical_block_id") or "").strip()
        if not block_id:
            continue
        entry = grouped.setdefault(block_id, {"block_id": block_id, "score": 0.0, "hits": 0})
        entry["score"] = max(float(entry["score"]), float(score))
        entry["hits"] = int(entry["hits"]) + 1
    ordered = sorted(grouped.values(), key=lambda item: (-item["hits"], -item["score"]))
    for entry in ordered:
        entry["score"] = round(float(entry["score"]), 4)
    return ordered[: max(1, int(limit))]
