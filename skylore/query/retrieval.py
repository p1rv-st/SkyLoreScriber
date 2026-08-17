"""`search_lore`: find prose by meaning rather than by name.

The other four tools all need a key the user already has -- a star name, a culture id.
This one answers "which cultures use a star to mark the rainy season", where the
question names nothing in the database and the answer is a sentence inside an article.

Two halves, fused. **BM25** over `sections_fts` matches words and needs no model; it is
strong in English, Russian and Spanish and weak in Chinese, because `unicode61` does not
segment CJK and a run of ideographs becomes a single token. **Dense** retrieval matches
meaning and is what carries Chinese and any paraphrase. Only the first exists today;
the second is behind `Embedder`, and everything below works with or without it, so the
BM25 baseline is measurable before any model is chosen.

Three things about this corpus shape the design:

**A section is four rows.** The same prose is materialised in en, ru, es and zh_CN, so
top-8 without deduplication is eight copies of two sections. Candidates are collapsed by
`(culture_id, ord)` *before* the cut, not after, or the cut silently loses variety.

**Matched in any language, served in one.** A Russian query should match Russian text
directly -- that is the whole payoff of materialising the po catalogues -- but the text
handed back is the English source, per `lang.prose_order`. Search language and output
language are different questions, and `lang` has kept them apart since step 3a.

**Whole sections, never chunks.** 1020 retrievable sections across four languages,
averaging 728 characters. Returning eight whole ones costs a few thousand tokens and
removes chunk-boundary loss entirely.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
from dataclasses import dataclass, field
from typing import Protocol

from .. import lang
from ..ingest import corpus
from . import cultures

# Rank-fusion constant, from the original RRF paper. Measured against the gold set at
# 10 and at 60: identical scores. It is not the parameter that matters here, so it keeps
# the published value rather than a tuned one.
RRF_K = 60

# Raised from 8 after measurement: 8 scored 6/10 on the retrieval questions and 12
# scored 8/10, at a cost of ~4.5k tokens rising to ~6.6k. The reason is understood
# rather than merely correlated -- these questions ask which *cultures* do something, so
# an answer needs several distinct ones, and eight slots under a per-culture cap cannot
# hold them. Treat as a floor, not a tuned optimum: ten questions cannot justify a
# finer number than this.
TOP_K = 12

# Candidates pulled from each ranker before fusion. Wider than TOP_K because fusion can
# promote something neither ranker had first.
CANDIDATES = 50

# Most sections one culture may take in an unfiltered result. Article length varies by
# more than two orders of magnitude -- lokono is 52k characters across 45 sections
# against a median culture's 5.7k -- so a long article crowds out every other tradition
# on any query, purely by having more chances to match. Measured: lokono took 5 of 8
# slots on a question about lost constellations.
#
# Applied only when the caller did not name a culture. "What does this culture say about
# X" is a legitimate question and wants every matching section it has; the cap is for
# the cross-culture case, which is what an unfiltered search means.
PER_CULTURE = 3


class Embedder(Protocol):
    """What the dense half needs. Deliberately tiny, so the runtime behind it --
    ONNX Runtime today, anything later -- stays an implementation detail."""

    name: str
    dim: int

    def encode_query(self, text: str) -> bytes: ...
    def encode_texts(self, texts: list[str]) -> list[bytes]: ...


@dataclass(frozen=True)
class Passage:
    """One retrieved section, in the language it will be read in."""
    section_id: int
    culture_id: str
    ord: int
    heading_path: str
    kind: str
    text: str
    lang: str
    matched_lang: str
    score: float
    references: dict[int, str] = field(default_factory=dict)

    @property
    def cross_language(self) -> bool:
        """True when the query matched one language and the text is served in another."""
        return self.matched_lang != self.lang


# ────────────────────────────────── vectors ──────────────────────────────────

def pack(vector: list[float]) -> bytes:
    """L2-normalise and store as float32, so cosine becomes a dot product."""
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return struct.pack(f"<{len(vector)}f", *(value / norm for value in vector))


def unpack(blob: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def dot(a: bytes, b: bytes) -> float:
    return sum(x * y for x, y in zip(unpack(a), unpack(b)))


# ─────────────────────────────────── bm25 ───────────────────────────────────

_WORD = re.compile(r"\w+", re.UNICODE)


def _fts_query(query: str) -> str:
    """A natural-language question as an FTS5 OR-query.

    FTS5 ANDs bare terms, which returns nothing for a sentence. Terms are ORed and each
    is quoted, so punctuation and words like `OR` are searched for rather than obeyed.
    IDF does the rest: common words match everything and are weighted down accordingly,
    so a stopword list would buy little and would have to exist per language.
    """
    terms = [term for term in _WORD.findall(query) if len(term) > 1]
    return " OR ".join(f'"{term}"' for term in terms)


def bm25(connection: sqlite3.Connection, query: str, *, langs: tuple[str, ...],
         culture: str | None = None, limit: int = CANDIDATES) -> list[tuple[int, float]]:
    """(section_id, rank) for the lexical half, best first."""
    expression = _fts_query(query)
    if not expression:
        return []
    placeholders = ",".join("?" * len(langs))
    sql = f"""
        SELECT s.id, bm25(sections_fts, 2.0, 1.0) AS score
          FROM sections_fts
          JOIN sections s ON s.id = sections_fts.rowid
         WHERE sections_fts MATCH ?
           AND s.retrievable = 1
           AND s.lang IN ({placeholders})
           {"AND s.culture_id = ?" if culture else ""}
         ORDER BY score
         LIMIT ?
    """
    params = [expression, *langs, *([culture] if culture else []), limit]
    return [(row[0], row[1]) for row in connection.execute(sql, params)]


# ─────────────────────────────────── dense ───────────────────────────────────

def embedded_langs(connection: sqlite3.Connection, model: str) -> tuple[str, ...]:
    """Which languages carry vectors for `model`, as recorded when they were built."""
    row = connection.execute(
        "SELECT languages FROM embedding_models WHERE model = ?", (model,)).fetchone()
    return tuple(json.loads(row[0])) if row else ()


def dense(connection: sqlite3.Connection, query: str, embedder: Embedder, *,
          langs: tuple[str, ...] | None = None, culture: str | None = None,
          limit: int = CANDIDATES) -> list[tuple[int, float]]:
    """(section_id, cosine) for the semantic half, best first.

    Brute force over the stored vectors. At ~1020 sections this needs no index and no
    FAISS: the whole comparison is a few thousand dot products.

    `langs` defaults to whatever was actually embedded, which is **not** the languages
    BM25 searches. Embedding all four made the model rank by language identity rather
    than by content: a Chinese query scored every Chinese section at 0.88 whatever it
    was about, with 0.003 between the best and the eighth. Embedding English alone --
    which a cross-lingual model can match a Chinese query against directly -- widened
    that spread to 0.026 and put the right section first. So the two halves differ on
    purpose: BM25 needs every language because it cannot cross them, and dense wants one
    because it can.
    """
    if langs is None:
        langs = embedded_langs(connection, embedder.name) or (lang.SOURCE_LANG,)
    probe = embedder.encode_query(query)
    placeholders = ",".join("?" * len(langs))
    sql = f"""
        SELECT e.section_id, e.vector
          FROM embeddings e
          JOIN sections s ON s.id = e.section_id
         WHERE e.model = ?
           AND s.retrievable = 1
           AND s.lang IN ({placeholders})
           {"AND s.culture_id = ?" if culture else ""}
    """
    params = [embedder.name, *langs, *([culture] if culture else [])]

    # A long section is several overlapping windows. The best window stands for the
    # section: max, not mean, because a section answers a question if *any* part of it
    # does, and averaging would penalise a long section for the paragraphs that are
    # about something else.
    best: dict[int, float] = {}
    for section_id, vector in connection.execute(sql, params):
        score = dot(probe, vector)
        if score > best.get(section_id, float("-inf")):
            best[section_id] = score

    scored = sorted(best.items(), key=lambda pair: -pair[1])
    return scored[:limit]


# ──────────────────────────────────── fusion ────────────────────────────────────

def rrf(*rankings: list[tuple[int, float]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion.

    Rank-based rather than score-based on purpose: a BM25 score and a cosine are not
    comparable numbers, and normalising them into agreement would invent a relationship
    that is not there. Rank is the one thing both rankers genuinely express.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for position, (section_id, _score) in enumerate(ranking):
            fused[section_id] = fused.get(section_id, 0.0) + 1.0 / (k + position + 1)
    return sorted(fused.items(), key=lambda pair: -pair[1])


# ──────────────────────────────────── search ────────────────────────────────────

def search_lore(
    connection: sqlite3.Connection,
    query: str,
    *,
    locale: str = lang.SOURCE_LANG,
    culture: str | None = None,
    limit: int = TOP_K,
    embedder: Embedder | None = None,
) -> list[Passage]:
    """Whole sections most likely to answer `query`, best first.

    Runs BM25 alone when no embedder is supplied, which is what makes the lexical
    baseline measurable before a model is chosen.
    """
    if not query.strip():
        return []

    available = lang.available_langs(connection)
    search_langs = lang.search_order(locale, available)
    output_order = lang.prose_order(locale, available)

    rankings = [bm25(connection, query, langs=search_langs, culture=culture)]
    if embedder is not None:
        # No `langs`: the dense half uses whatever was embedded, which is deliberately
        # narrower than what BM25 searches. See `dense`.
        rankings.append(dense(connection, query, embedder, culture=culture))

    fused = rrf(*rankings)
    if not fused:
        return []

    rows = _section_rows(connection, [section_id for section_id, _ in fused])

    # Collapse the four language rows of one section before the cut, not after: eight
    # results would otherwise be two sections in four languages each.
    best: dict[tuple[str, int], tuple[float, str]] = {}
    for section_id, score in fused:
        row = rows.get(section_id)
        if row is None:
            continue
        key = (row["culture_id"], row["ord"])
        if key not in best:
            best[key] = (score, row["lang"])

    ranked = sorted(best.items(), key=lambda item: -item[1][0])
    if culture is None:
        ranked = _cap_per_culture(ranked, PER_CULTURE)
    ordered = ranked[:limit]
    return [
        passage for passage in (
            _passage(connection, culture_id, ordinal, matched_lang, score, output_order)
            for (culture_id, ordinal), (score, matched_lang) in ordered
        ) if passage is not None
    ]


def _cap_per_culture(ranked: list, cap: int) -> list:
    """Keep rank order, but let no culture take more than `cap` of the result.

    Overflow is dropped rather than pushed to the end: a ninth lokono section is not a
    better answer than the tenth-ranked one from somewhere else, and keeping it would
    only move the crowding further down.
    """
    counts: dict[str, int] = {}
    kept = []
    for item in ranked:
        (culture_id, _ordinal), _payload = item
        if counts.get(culture_id, 0) >= cap:
            continue
        counts[culture_id] = counts.get(culture_id, 0) + 1
        kept.append(item)
    return kept


def _section_rows(connection: sqlite3.Connection,
                  section_ids: list[int]) -> dict[int, dict]:
    if not section_ids:
        return {}
    placeholders = ",".join("?" * len(section_ids))
    return {
        row[0]: {"culture_id": row[1], "ord": row[2], "lang": row[3]}
        for row in connection.execute(
            f"SELECT id, culture_id, ord, lang FROM sections WHERE id IN ({placeholders})",
            section_ids,
        )
    }


def _passage(connection: sqlite3.Connection, culture_id: str, ordinal: int,
             matched_lang: str, score: float,
             output_order: tuple[str, ...]) -> Passage | None:
    """One matched section, resolved into the language it should be read in."""
    variants = {
        row[0]: row for row in connection.execute(
            "SELECT lang, id, heading_path, kind, text FROM sections"
            " WHERE culture_id = ? AND ord = ?", (culture_id, ordinal))
    }
    images_usable, = connection.execute(
        "SELECT images_usable FROM cultures WHERE id = ?", (culture_id,)).fetchone()
    excluded = cultures.excluded_paths(connection, culture_id)

    for language in output_order:
        row = variants.get(language)
        if row is None or not row[4].strip():
            continue
        text, _ = cultures.strip_unservable_images(
            row[4], images_usable=bool(images_usable), excluded=excluded)
        refs = {
            number: text_
            for number, text_ in _references(connection, culture_id, language).items()
            if number in corpus.cited_refs(text)
        }
        return Passage(
            section_id=row[1], culture_id=culture_id, ord=ordinal,
            heading_path=row[2], kind=row[3], text=text, lang=language,
            matched_lang=matched_lang, score=score, references=refs,
        )
    return None


def _references(connection: sqlite3.Connection, culture_id: str,
                language: str) -> dict[int, str]:
    return {
        row[0]: row[1] for row in connection.execute(
            "SELECT ref_num, text FROM section_refs WHERE culture_id = ? AND lang = ?",
            (culture_id, language))
    }
