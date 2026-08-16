"""The culture catalogue and whole articles.

`find_cultures` does no searching. With 34 cultures the entire catalogue is ~16k
characters, so a directory the model reads beats a similarity search it has to be
trusted to get right. The optional `query` narrows that list by substring; it is a
convenience, not retrieval, and it is documented as such so nobody later mistakes a
missing row for a ranking failure.

`get_culture_article` returns the article whole rather than chunked. The corpus can
afford it -- ~5700 characters per culture on average -- and handing the model a
complete article removes chunk-boundary loss entirely, making the agent pattern
*locate, then read in full*.

Two things this module is responsible for that are easy to miss:

**Citations.** Sections store `[#7]` verbatim rather than substituting the source
inline, so the database keeps fidelity to what upstream wrote. Composing the citation
is this layer's job; without it the model receives a bare marker and the reference is
lost.

**Images it is not allowed to serve.** The prose carries inline markdown image
references, and one of them -- `japanese_moon_stations/chart.webp` -- is the single
row in `excluded_assets`, admitted on the explicit condition that the map is neither
ingested nor served. It is present in all four languages of that culture's text.
Returning the article verbatim would hand the model a pointer to it, so references
are stripped here, under two independent rules (see `strip_unservable_images`).
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field

from . import corpus, lang

# `![alt](path)` in either of the two forms the corpus uses, with or without alt text.
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

# Boilerplate. Kept in the database for provenance, excluded from an article body:
# `attribution` and `text_licenses` carry the same information structurally, and the
# prose copies look alike across every culture.
_BOILERPLATE_KINDS = {"authors", "license"}


@dataclass(frozen=True)
class CultureCard:
    """One row of the catalogue."""
    id: str
    name: lang.Resolved | None
    region: str | None
    classification: list[str]
    summary: lang.Resolved | None
    constellation_count: int
    native_lang: str | None
    images_usable: bool
    text_licenses: list[str]
    attribution: str


@dataclass(frozen=True)
class Section:
    ord: int
    kind: str
    level: int
    heading: str | None
    heading_path: str
    text: str
    lang: str
    # Set when this row fell back at ingest time because `lang` was missing upstream.
    # Distinct from the language resolution done here, and both must stay visible:
    # an answer may not present either kind of fallback as a translation.
    fallback_from: str | None
    constellation_id: str | None
    references: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Article:
    culture_id: str
    name: lang.Resolved | None
    locale: str
    sections: list[Section]
    attribution: str
    text_licenses: list[str]
    images_usable: bool
    # Image references removed before returning, and why. Reported rather than
    # silently dropped: a caller comparing the article against the upstream markdown
    # should be able to see that the difference is deliberate.
    omitted_images: list[tuple[str, str]] = field(default_factory=list)


# ─────────────────────────────────── licensing ───────────────────────────────────

def excluded_paths(connection: sqlite3.Connection, culture_id: str) -> set[str]:
    return {
        row[0] for row in connection.execute(
            "SELECT path FROM excluded_assets WHERE culture_id = ?", (culture_id,)
        )
    }


def strip_unservable_images(
    text: str, *, images_usable: bool, excluded: set[str]
) -> tuple[str, list[tuple[str, str]]]:
    """Remove image references the licence does not let us serve.

    Two independent rules, deliberately not collapsed into one. Artwork licences
    differ from prose licences, so a culture may permit the text while permitting
    nothing of the images (`images_usable = 0`); and a specific file may be carved
    out by a manual review even where the culture's artwork is otherwise fine
    (`excluded_assets`). Today both rules happen to catch the same file. They will
    not always, and a single check would quietly stop covering one of the cases.

    Alt text and the caption prose around an image are left alone -- they are the
    article's own words, and removing them would damage the text to no licensing
    purpose.
    """
    omitted: list[tuple[str, str]] = []

    def replace(match: re.Match) -> str:
        path = match.group(2)
        if path in excluded:
            omitted.append((path, "excluded_assets"))
            return ""
        if not images_usable:
            omitted.append((path, "images_usable = 0"))
            return ""
        return match.group(0)

    stripped = _IMAGE.sub(replace, text)
    # Collapse the blank line an image occupying its own paragraph leaves behind.
    return re.sub(r"\n{3,}", "\n\n", stripped).strip(), omitted


# ─────────────────────────────────── catalogue ───────────────────────────────────

def find_cultures(
    connection: sqlite3.Connection,
    *,
    query: str | None = None,
    region: str | None = None,
    locale: str = lang.SOURCE_LANG,
) -> list[CultureCard]:
    """The catalogue, whole. `query` and `region` narrow it; neither ranks it.

    The name resolves through `lang.name_order` and the summary through
    `lang.prose_order`, so a card can legitimately carry a Russian name above an
    English summary. That is the rule applied consistently, not an oversight: a name
    is a term the corpus already holds in the user's language, while prose is
    translated downstream anyway and loses least when sent as the source.
    """
    available = lang.available_langs(connection)
    names = _by_culture(connection, "SELECT culture_id, lang, value FROM culture_names")
    summaries = _by_culture(
        connection, "SELECT culture_id, lang, summary FROM culture_summaries"
    )

    cards: list[CultureCard] = []
    for row in connection.execute("""
        SELECT id, region, classification, native_lang, constellation_count,
               images_usable, text_licenses, attribution
          FROM cultures ORDER BY id
    """):
        (culture_id, culture_region, classification, native_lang,
         count, images_usable, text_licenses, attribution) = row
        if region is not None and culture_region != region:
            continue
        if query and not _matches(query, names.get(culture_id, {}),
                                  summaries.get(culture_id, {}), culture_id):
            continue
        cards.append(CultureCard(
            id=culture_id,
            name=lang.pick(names.get(culture_id, {}),
                           lang.name_order(locale, available), locale),
            region=culture_region,
            classification=json.loads(classification or "[]"),
            summary=lang.pick(summaries.get(culture_id, {}),
                              lang.prose_order(locale, available), locale),
            constellation_count=count,
            native_lang=native_lang,
            images_usable=bool(images_usable),
            text_licenses=json.loads(text_licenses or "[]"),
            attribution=attribution,
        ))
    return cards


def _by_culture(connection: sqlite3.Connection, sql: str) -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = {}
    for culture_id, value_lang, value in connection.execute(sql):
        table.setdefault(culture_id, {})[value_lang] = value
    return table


def _matches(query: str, names: dict[str, str], summaries: dict[str, str],
             culture_id: str) -> bool:
    """Substring over every language's name and summary, plus the id.

    Every language, not the requested one: someone narrowing the catalogue with an
    English word should still find a culture they went on to read in Russian.
    """
    needle = query.casefold()
    haystack = [culture_id, *names.values(), *summaries.values()]
    return any(needle in text.casefold() for text in haystack)


# ──────────────────────────────────── articles ────────────────────────────────────

def get_culture_article(
    connection: sqlite3.Connection,
    culture_id: str,
    *,
    locale: str = lang.SOURCE_LANG,
    section: str | None = None,
    include_boilerplate: bool = False,
) -> Article | None:
    """One culture's article, whole and in reading order.

    `section` takes a `kind` ("description", "constellations", …) or a heading path,
    matched case-insensitively. It exists for `lokono`, which is 52k characters --
    six times the next largest article -- and is the one culture where reading the
    whole thing is a real cost.

    The language resolves through `lang.prose_order`, which puts the source first.
    Against the current corpus that means every article comes back in English for
    every locale, because English is total over `sections`. `Section.lang` records
    what was actually served, so the caller never has to assume.
    """
    row = connection.execute(
        "SELECT attribution, text_licenses, images_usable FROM cultures WHERE id = ?",
        (culture_id,),
    ).fetchone()
    if row is None:
        return None
    attribution, text_licenses, images_usable = row
    images_usable = bool(images_usable)

    available = lang.available_langs(connection)
    order = lang.prose_order(locale, available)
    excluded = excluded_paths(connection, culture_id)
    references = _references(connection, culture_id)

    names = _by_culture(
        connection, "SELECT culture_id, lang, value FROM culture_names"
    ).get(culture_id, {})

    sections: list[Section] = []
    omitted: list[tuple[str, str]] = []
    for ordinal in _section_ordinals(connection, culture_id, order):
        built = _section(connection, culture_id, ordinal, order, references,
                         images_usable=images_usable, excluded=excluded)
        if built is None:
            continue
        if not include_boilerplate and built.kind in _BOILERPLATE_KINDS:
            continue
        if section is not None and not _wanted(built, section):
            continue
        sections.append(built)
        omitted.extend(_omitted_for(built, connection, culture_id, order,
                                    images_usable, excluded))

    return Article(
        culture_id=culture_id,
        name=lang.pick(names, lang.name_order(locale, available), locale),
        locale=locale,
        sections=sections,
        attribution=attribution,
        text_licenses=json.loads(text_licenses or "[]"),
        images_usable=images_usable,
        omitted_images=omitted,
    )


def _wanted(section: Section, wanted: str) -> bool:
    """Whether a section is the one asked for.

    A `kind`, a whole heading path, or a path prefix -- "Description" selects that
    section and everything nested under it. Deliberately not a substring match:
    asking for "introduction" would otherwise also return
    "Description › Lokono astronomical knowledge: an introduction", and a selector
    that quietly returns neighbouring prose is worse than one that returns nothing.
    """
    needle = wanted.casefold().strip()
    path = section.heading_path.casefold()
    return (section.kind.casefold() == needle
            or path == needle
            or path.startswith(f"{needle} › "))


def _section_ordinals(connection: sqlite3.Connection, culture_id: str,
                      order: tuple[str, ...]) -> list[int]:
    """Reading order, taken from whichever language leads and actually has rows.

    `sections.ord` is per (culture, lang), and the translations can diverge from the
    source in how many subsections they split into (see `corpus.localise_section`),
    so the sequence has to come from one language rather than be merged across them.
    """
    for language in order:
        rows = connection.execute(
            "SELECT ord FROM sections WHERE culture_id = ? AND lang = ? ORDER BY ord",
            (culture_id, language),
        ).fetchall()
        if rows:
            return [row[0] for row in rows]
    return []


def _section(connection: sqlite3.Connection, culture_id: str, ordinal: int,
             order: tuple[str, ...], references: dict[str, dict[int, str]],
             *, images_usable: bool, excluded: set[str]) -> Section | None:
    """One section, resolved per field rather than per article.

    A culture whose Russian translation is missing one subsection must not drop the
    whole article to English -- only that subsection.
    """
    rows = {
        row[0]: row for row in connection.execute(
            "SELECT lang, kind, level, heading, heading_path, text, fallback_from,"
            "       constellation_id"
            "  FROM sections WHERE culture_id = ? AND ord = ?",
            (culture_id, ordinal),
        )
    }
    for language in order:
        row = rows.get(language)
        if row is None or not row[5].strip():
            continue
        _, kind, level, heading, heading_path, text, fallback_from, constellation_id = row
        text, _ = strip_unservable_images(text, images_usable=images_usable,
                                          excluded=excluded)
        return Section(
            ord=ordinal,
            kind=kind,
            level=level,
            heading=heading,
            heading_path=heading_path,
            text=text,
            lang=language,
            fallback_from=fallback_from,
            constellation_id=constellation_id,
            references=_cited(text, kind, references.get(language, {})),
        )
    return None


def _omitted_for(section: Section, connection: sqlite3.Connection, culture_id: str,
                 order: tuple[str, ...], images_usable: bool,
                 excluded: set[str]) -> list[tuple[str, str]]:
    """What `_section` stripped, recomputed against the raw text for reporting."""
    raw = connection.execute(
        "SELECT text FROM sections WHERE culture_id = ? AND ord = ? AND lang = ?",
        (culture_id, section.ord, section.lang),
    ).fetchone()
    if raw is None:
        return []
    _, omitted = strip_unservable_images(raw[0], images_usable=images_usable,
                                         excluded=excluded)
    return omitted


def _references(connection: sqlite3.Connection,
                culture_id: str) -> dict[str, dict[int, str]]:
    table: dict[str, dict[int, str]] = {}
    for ref_lang, ref_num, text in connection.execute(
        "SELECT lang, ref_num, text FROM section_refs WHERE culture_id = ?",
        (culture_id,),
    ):
        table.setdefault(ref_lang, {})[ref_num] = text
    return table


def _cited(text: str, kind: str, available: dict[int, str]) -> dict[int, str]:
    """The `[#N]` markers this passage uses, resolved to their sources.

    The References section is skipped: its markers are the definitions themselves, so
    resolving them would pair every entry with a copy of itself.
    """
    if kind == "references":
        return {}
    return {
        number: available[number]
        for number in sorted(corpus.cited_refs(text))
        if number in available
    }
