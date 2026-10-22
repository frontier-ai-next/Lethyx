"""Universal, script-agnostic text utilities for lexical memory retrieval.

There are deliberately no hard-coded alphabets, character-code ranges,
per-language stopword lists, or regex character classes in this module.
Token boundaries come from Unicode properties (``str.isalnum``) and lexical
scoring comes from character n-grams over the normalized alphanumeric stream.
The same code therefore works for Latin, Cyrillic, Greek, Arabic, Hebrew,
Devanagari, Han, Kana, Hangul, and mixed-script text.

Why n-grams instead of words for similarity
-------------------------------------------
Scripts written without spaces cannot be split into words without a
morphological dictionary. Character n-grams need no dictionary, are
language-neutral, tolerate inflection (Russian, German, Arabic), and match
substrings inside spaceless scripts (Japanese, Chinese, Thai). That is the
universal primitive; word tokens are only used for display and candidate
term reporting.

Stopwords are learned, not declared
-----------------------------------
:func:`rank` weights query terms by inverse document frequency computed from
the candidate corpus itself. Terms that appear in most records therefore
carry little weight without anyone maintaining a list for a particular
language.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence

#: N-gram width used for lexical similarity. Three is the smallest width that
#: is discriminative for inflected and spaceless scripts without exploding the
#: n-gram set for long records.
NGRAM_SIZE = 3

#: A query term that also appears in a record title is a strong routing
#: signal. The multiplier is language-neutral and does not special-case any
#: alphabet; it only distinguishes the human-authored title from body text.
TITLE_WEIGHT = 1.75


def normalize(value: object) -> str:
    """Return NFKC-normalized, case-folded, whitespace-collapsed text."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.casefold().split())


def tokens(value: object) -> list[str]:
    """Split text on every character that is not a Unicode letter or digit."""

    result: list[str] = []
    current: list[str] = []
    for char in normalize(value):
        if char.isalnum():
            current.append(char)
        elif current:
            result.append("".join(current))
            current = []
    if current:
        result.append("".join(current))
    return result


def stream(value: object) -> str:
    """Return the alphanumeric character stream used for n-gram indexing."""

    return "".join(tokens(value))


def grams(value: object, size: int = NGRAM_SIZE) -> set[str]:
    """Return the set of character n-grams of the alphanumeric stream."""

    text = stream(value)
    if not text:
        return set()
    if len(text) <= size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def containment(needle: object, haystack: object, size: int = NGRAM_SIZE) -> float:
    """Fraction of ``needle`` n-grams present in ``haystack`` (0..1)."""

    needle_grams = grams(needle, size)
    if not needle_grams:
        return 0.0
    return len(needle_grams & grams(haystack, size)) / len(needle_grams)


def similarity(left: object, right: object, size: int = NGRAM_SIZE) -> float:
    """Symmetric Jaccard similarity of character n-grams (0..1)."""

    left_grams = grams(left, size)
    right_grams = grams(right, size)
    if not left_grams and not right_grams:
        return 1.0 if normalize(left) == normalize(right) else 0.0
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def distinctive_terms(value: object, limit: int | None = None) -> list[str]:
    """Unique query terms, longest first, with no declared stopword list."""

    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens(value):
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    ordered.sort(key=lambda token: (-len(token), token))
    return ordered[:limit] if limit else ordered


def _occurs(token: str, document_tokens: Sequence[str], document_stream: str) -> bool:
    """Return whether a query term occurs in a document.

    Exact token matches are preferred. Longer terms also match by n-gram
    containment so inflected forms (Russian, German, Arabic) and spaceless
    compounds (Japanese) still count as the same term.
    """

    if token in document_tokens:
        return True
    if len(token) < 4:
        return False
    return containment(token, document_stream) >= 0.55


def rank(
    query: object,
    documents: Sequence[object],
    *,
    titles: Sequence[object] | None = None,
    limit: int | None = None,
    min_relative: float = 0.5,
) -> list[tuple[int, float, list[str]]]:
    """Rank documents against a query with corpus-derived term weights.

    Returns ``(document_index, relative_score, matched_terms)`` tuples, best
    first. ``relative_score`` is normalized so the best match is ``1.0``.

    Weights are inverse document frequencies computed from ``documents``, so a
    term that occurs in most records is automatically treated as glue without
    a hand-written stopword list for any language. When ``titles`` is supplied,
    a query term that appears in the title contributes more than the same term
    buried in body text; this is what makes "where is the config" prefer the
    record titled "Config locations" over a long document that merely mentions
    config often.

    A document must match at least one discriminating term. Terms that occur in
    most of the corpus are not allowed to carry a match by themselves, which is
    what keeps a question such as "what is the capital of France" from matching
    every English record on "is" and "the". A corpus-common term may still
    count when it appears in the title, because a title is an explicit routing
    signal (for example a product name such as "opencode").
    """

    query_terms = list(dict.fromkeys(tokens(query)))
    if not query_terms or not documents:
        return []
    title_values = list(titles) if titles is not None and len(titles) == len(documents) else []
    title_tokens = [tokens(title) for title in title_values]
    title_streams = [stream(title) for title in title_values]
    document_tokens = [tokens(document) for document in documents]
    document_streams = [stream(document) for document in documents]
    total = len(documents)

    df: dict[str, int] = {}
    for term in query_terms:
        df[term] = sum(
            1
            for tokens_for_doc, stream_for_doc in zip(document_tokens, document_streams)
            if _occurs(term, tokens_for_doc, stream_for_doc)
        )
    weights = {term: math.log((total + 1) / (df[term] + 1)) + 1.0 for term in query_terms}

    ranked: list[tuple[int, float, list[str]]] = []
    # A term that occurs in more than a third of the corpus is glue. The
    # threshold is deliberately stricter than "half": on a small corpus half
    # is a single record, and one extra record would turn a common word into
    # a signal. A title match still rescues a term, because a title is an
    # explicit routing signal.
    common_df_limit = max(1, total // 3)
    very_common_df_limit = max(2, total - 1)
    for index, (tokens_for_doc, stream_for_doc) in enumerate(
        zip(document_tokens, document_streams)
    ):
        matched = [term for term in query_terms if _occurs(term, tokens_for_doc, stream_for_doc)]
        if not matched:
            continue
        scored_terms: list[str] = []
        raw = 0.0
        for term in matched:
            title_hit = bool(
                title_tokens and _occurs(term, title_tokens[index], title_streams[index])
            )
            if df[term] >= very_common_df_limit:
                continue
            if df[term] > common_df_limit and not title_hit:
                continue
            scored_terms.append(term)
            raw += weights[term] * (TITLE_WEIGHT if title_hit else 1.0)
        if not scored_terms:
            continue
        ranked.append((index, raw, scored_terms))

    if not ranked:
        return []
    ranked.sort(key=lambda item: (-item[1], item[0]))
    best = ranked[0][1] or 1.0
    floor = best * max(0.0, min(1.0, min_relative))
    normalized = [
        (index, round(raw / best, 4), matched) for index, raw, matched in ranked if raw >= floor
    ]
    return normalized[:limit] if limit else normalized
