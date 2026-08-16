"""Name lookup: turn whatever the user typed into sky objects.

`lookup_star` has to accept every key someone might hold -- an HIP number, an
international name, a Bayer designation, a native name from any of 34 cultures, or a
gloss in any of four languages -- and `find_constellation` the same minus the HIP.
That is a UNION over `stars` and `names` with fuzzy matching, not one lookup path,
and the shape of this module follows from three properties of the data.

**Matching is tiered, because substring matching alone ranks badly.** "Net" is the
exact name of 7 objects and a substring of 132 rows. Trigram FTS cannot tell those
apart -- it has no notion of relevance -- so the tier is computed here and does the
ordering that BM25 would do for prose.

**Trigram FTS matches nothing below three characters.** 1222 name rows are shorter
than that, 237 of them two-character CJK, so `毕宿` is unreachable through
`names_fts` no matter how it is queried. Short queries fall back to `LIKE '%q%'`,
a trivial scan at 14731 rows. Skipping this reads as "search does not work in
Chinese".

**A match is not an answer.** What matched is one row; what the caller wants is the
object, with every name it has in every language, so the answering model can choose.
`NameSet` is that, and it keeps `native` and `pronounce` untranslated and separate
from the glosses -- 毕宿 is the name, not English awaiting rendering.

Every response carries `cultures.attribution` for each culture it drew on. That is a
licence condition, not a courtesy (PLAN.md §3).
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field

from . import cultures, lang

# Below this, a query forms no trigram and `names_fts` returns nothing.
MIN_TRIGRAM = 3

# "HIP 21421", "hip21421", "21421" -- all the same star.
_HIP = re.compile(r"^(?:hip\s*)?(\d{1,6})$", re.I)

# Match quality, best first. Ordering happens on these before anything else.
TIERS = ("hip", "exact", "prefix", "substring")


@dataclass(frozen=True)
class Match:
    """One `names` row that matched, with why it matched."""
    name_id: int
    value: str
    kind: str                    # native | pronounce | gloss
    lang: str | None             # NULL for native and pronounce, by schema CHECK
    culture_id: str | None       # NULL for international names
    constellation_id: str | None
    hip: int | None
    tier: str

    @property
    def is_star(self) -> bool:
        return self.hip is not None


@dataclass(frozen=True)
class NameSet:
    """Every name one culture has for one object, and the one to use in prose.

    The whole dictionary travels, not just the preferred string. Russian gloss
    coverage is 4% on stars, so a single resolved name would be English passed off
    as Russian in almost every lookup; returning all of them is what makes the gap
    visible to the model instead of silent (TECHDEBT.md §1).
    """
    culture_id: str | None
    native: str | None
    pronounce: str | None
    glosses: dict[str, str]      # lang -> value
    preferred: lang.Resolved | None

    @property
    def display(self) -> str | None:
        """The name to write in prose: the resolved gloss, else the native name.

        Falling through to `native` only when no gloss resolves is deliberate. A
        Chinese speaker asking about a Norse figure wants the Chinese gloss with the
        Norse name beside it, not the Norse name as the answer.
        """
        return self.preferred.value if self.preferred else self.native


@dataclass(frozen=True)
class Star:
    hip: int
    iau_name: str | None
    designation: str | None
    names: list[NameSet] = field(default_factory=list)      # one per culture that names it
    figures: list[Constellation] = field(default_factory=list)  # cultures that draw through it


@dataclass(frozen=True)
class Prose:
    """Prose about one constellation, and which upstream file it came from.

    Two sources land here and they are not interchangeable. `index` is the
    `description` field of `index.json` -- a one-line note, 7 to 130 characters.
    `article` is a `##### <native name>` subsection of `description.md`, up to 2104.
    Only 51 of 1529 constellations have either, and just one has both, so a caller
    that assumed a single source would silently lose most of it.
    """
    text: str
    lang: str
    source: str                  # "index" | "article"
    heading_path: str | None     # article sections only
    section_ord: int | None


@dataclass(frozen=True)
class Constellation:
    id: str
    culture_id: str
    iau: str | None
    names: NameSet
    attribution: str
    # Populated for detail results. Empty in `Star.figures`, which is a list view:
    # Aldebaran alone is drawn by 27 cultures, and prose for each would swamp the
    # answer it is meant to support.
    prose: list[Prose] = field(default_factory=list)


# ─────────────────────────────────── matching ───────────────────────────────────

def fold(text: str) -> str:
    """Compare-form of a name: decomposed, unaccented, case-insensitive.

    The corpus is inconsistently normalised -- 1006 name rows are NFC and 18 are NFD,
    the latter all Bugis and Mandar. A user types NFC, because that is what keyboards
    and web forms produce, so `Bintoѐng Bola Kѐppang` typed normally never equalled the
    decomposed value stored for it, and the name was unreachable.

    Decomposing then dropping combining marks lands both forms on the same string, and
    matches what `names_fts` already does: its tokenizer folds diacritics, which is why
    the index found this row while exact matching did not.
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    ).casefold()


def _normal_forms(query: str) -> list[str]:
    """The Unicode normal forms of a query, deduplicated, original first.

    Usually one string; two when the query carries a mark that composes, which is the
    only case where the index and the query can disagree.
    """
    forms = [query]
    for form in ("NFC", "NFD"):
        candidate = unicodedata.normalize(form, query)
        if candidate not in forms:
            forms.append(candidate)
    return forms


def _register_fold(connection: sqlite3.Connection) -> None:
    """Make `fold` callable from SQL. Idempotent, and cheap enough to do per query."""
    connection.create_function("fold", 1, fold, deterministic=True)


def _fts_query(query: str) -> str:
    """A trigram substring search for `query`, safe against FTS5 syntax.

    Everything is one quoted phrase, so operators the user typed (`OR`, `*`, `-`)
    are searched for rather than obeyed. Internal quotes are doubled, per FTS5.
    """
    return '"' + query.replace('"', '""') + '"'


def _like_pattern(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _prefix_pattern(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


def parse_hip(query: str) -> int | None:
    """The HIP number in a query, if it is one. `None` otherwise."""
    match = _HIP.match(query.strip())
    return int(match.group(1)) if match else None


def search(
    connection: sqlite3.Connection,
    query: str,
    *,
    locale: str = lang.SOURCE_LANG,
    subject: str | None = None,      # "star" | "constellation" | None for both
    culture: str | None = None,
    limit: int = 50,
) -> list[Match]:
    """Name rows matching `query`, best match first.

    Ordering is tier, then language priority from `lang.search_order`, then the
    source's own `rank` (0 is an object's primary name). Native and pronounce rows
    carry no language and sort with the requested one: they are the object's own
    name, not a translation competing with it.
    """
    query = query.strip()
    if not query:
        return []
    _register_fold(connection)

    # The trigram tokenizer strips combining marks but does not decompose precomposed
    # characters -- and only folds precomposed Latin, not Cyrillic. So an NFD query
    # reaches NFD-stored names (Bugis, Mandar) and an NFC query reaches NFC-stored ones
    # (everything else), and neither reaches both. The corpus is mixed, 1006 rows NFC
    # against 18 NFD, so the index is searched in every form the query has.
    forms = _normal_forms(query)

    # Below three characters the FTS index cannot help; scan instead. The table is
    # small enough that this costs nothing, and without it every two-character CJK
    # name in the corpus is unreachable. `fold` on both sides makes that path
    # normalisation-insensitive on its own, so it needs no such union.
    if len(query) >= MIN_TRIGRAM:
        source = " UNION ".join(
            f"SELECT rowid AS id FROM names_fts WHERE names_fts MATCH :fts{index}"
            for index in range(len(forms))
        )
    else:
        source = "SELECT id FROM names WHERE fold(value) LIKE :like ESCAPE '\\'"

    order = lang.search_order(locale, lang.available_langs(connection))
    # A searched CASE rather than `CASE n.lang WHEN ...`: native and pronounce rows
    # have lang NULL, and `NULL = NULL` is never true, so a simple CASE would drop
    # them to the bottom. They belong at the top -- an object's own name is not a
    # translation competing with the requested language.
    priority = "CASE WHEN n.lang IS NULL THEN 0 " + " ".join(
        f"WHEN n.lang = '{tag}' THEN {index + 1}" for index, tag in enumerate(order)
    ) + " ELSE 99 END"

    rows = connection.execute(f"""
        WITH candidates AS ({source})
        SELECT n.id, n.value, n.kind, n.lang, n.culture_id, n.constellation_id, n.hip,
               CASE WHEN fold(n.value) = :q                     THEN 'exact'
                    WHEN fold(n.value) LIKE :prefix ESCAPE '\\'  THEN 'prefix'
                    ELSE 'substring' END AS tier
          FROM candidates c
          JOIN names n ON n.id = c.id
         WHERE (:subject IS NULL
                OR (:subject = 'star' AND n.hip IS NOT NULL)
                OR (:subject = 'constellation' AND n.constellation_id IS NOT NULL))
           AND (:culture IS NULL OR n.culture_id = :culture)
         ORDER BY CASE tier WHEN 'exact' THEN 0 WHEN 'prefix' THEN 1 ELSE 2 END,
                  {priority},
                  n.rank,
                  n.id
         LIMIT :limit
    """, {
        **{f"fts{index}": _fts_query(form) for index, form in enumerate(forms)},
        "like": _like_pattern(fold(query)),
        "prefix": _prefix_pattern(fold(query)),
        "q": fold(query),
        "subject": subject,
        "culture": culture,
        "limit": limit,
    }).fetchall()

    return [Match(*row) for row in rows]


# ──────────────────────────────── name dictionaries ────────────────────────────────

def _nameset(rows: list[tuple], culture_id: str | None, locale: str,
             available: tuple[str, ...]) -> NameSet:
    """Build one culture's NameSet from its `names` rows for a single object."""
    native = pronounce = None
    glosses: dict[str, str] = {}
    for kind, name_lang, value, _rank in rows:
        if kind == "gloss":
            glosses.setdefault(name_lang, value)
        elif kind == "native" and native is None:
            native = value
        elif kind == "pronounce" and pronounce is None:
            pronounce = value
    return NameSet(
        culture_id=culture_id,
        native=native,
        pronounce=pronounce,
        glosses=glosses,
        preferred=lang.pick(glosses, lang.name_order(locale, available), locale),
    )


def _grouped_names(connection: sqlite3.Connection, where: str, params: dict,
                   locale: str) -> list[NameSet]:
    """NameSets for one object, one per culture that names it."""
    available = lang.available_langs(connection)
    rows = connection.execute(
        f"SELECT culture_id, kind, lang, value, rank FROM names WHERE {where} "
        f"ORDER BY culture_id IS NOT NULL, culture_id, rank, id", params
    ).fetchall()

    by_culture: dict[str | None, list[tuple]] = {}
    for culture_id, kind, name_lang, value, rank in rows:
        by_culture.setdefault(culture_id, []).append((kind, name_lang, value, rank))
    return [_nameset(entries, culture_id, locale, available)
            for culture_id, entries in by_culture.items()]


# ───────────────────────────────────── tools ─────────────────────────────────────

def lookup_star(
    connection: sqlite3.Connection,
    query: str,
    *,
    locale: str = lang.SOURCE_LANG,
    limit: int = 10,
) -> list[Star]:
    """Stars matching `query`, with every culture that names or draws each one.

    Accepts an HIP number, an international name, a Bayer designation, a native name
    from any culture, or a gloss in any language. A bare number is treated as an HIP
    and short-circuits the text search: it is unambiguous, and searching for "21421"
    as a substring would return unrelated names that happen to contain it.
    """
    hip = parse_hip(query)
    if hip is not None:
        hips = [hip] if _star_exists(connection, hip) else []
    else:
        hips = _ordered_unique(
            match.hip for match in search(connection, query, locale=locale,
                                          subject="star", limit=limit * 5)
        )[:limit]
        if not hips:
            hips = _designation_hits(connection, query, limit)

    return [_star(connection, h, locale) for h in hips]


def find_constellation(
    connection: sqlite3.Connection,
    query: str,
    *,
    locale: str = lang.SOURCE_LANG,
    culture: str | None = None,
    limit: int = 10,
) -> list[Constellation]:
    """Constellations whose name matches `query`, in any culture and any language.

    A detail view, so each result carries its prose as well as its names.
    """
    ids = _ordered_unique(
        match.constellation_id
        for match in search(connection, query, locale=locale, subject="constellation",
                            culture=culture, limit=limit * 5)
    )[:limit]
    return [_constellation(connection, cid, locale, with_prose=True) for cid in ids]


# ──────────────────────────────────── internals ────────────────────────────────────

def _ordered_unique(values) -> list:
    """De-duplicate while keeping first-seen order -- the ranking `search` produced."""
    seen: dict = {}
    for value in values:
        if value is not None:
            seen.setdefault(value, None)
    return list(seen)


def _star_exists(connection: sqlite3.Connection, hip: int) -> bool:
    return connection.execute("SELECT 1 FROM stars WHERE hip = ?", (hip,)).fetchone() is not None


def _designation_hits(connection: sqlite3.Connection, query: str, limit: int) -> list[int]:
    """Bayer/Flamsteed designations, which live in `stars` rather than in `names`.

    Only consulted when the name search found nothing, so "alf And" resolves without
    designations competing with real names for the top of every result list.
    """
    rows = connection.execute(
        "SELECT hip FROM stars WHERE designation LIKE ? ESCAPE '\\' "
        "ORDER BY designation = ? COLLATE NOCASE DESC, hip LIMIT ?",
        (_like_pattern(query), query, limit),
    ).fetchall()
    return [row[0] for row in rows]


def _star(connection: sqlite3.Connection, hip: int, locale: str) -> Star:
    iau_name, designation = connection.execute(
        "SELECT iau_name, designation FROM stars WHERE hip = ?", (hip,)
    ).fetchone()
    figures = [
        _constellation(connection, row[0], locale)
        for row in connection.execute(
            "SELECT constellation_id FROM star_figures WHERE hip = ? ORDER BY culture_id",
            (hip,),
        )
    ]
    return Star(
        hip=hip,
        iau_name=iau_name,
        designation=designation,
        names=_grouped_names(connection, "hip = :hip", {"hip": hip}, locale),
        figures=figures,
    )


def _constellation(connection: sqlite3.Connection, constellation_id: str,
                   locale: str, *, with_prose: bool = False) -> Constellation:
    culture_id, iau, attribution, images_usable = connection.execute(
        "SELECT c.culture_id, c.iau, k.attribution, k.images_usable FROM constellations c "
        "JOIN cultures k ON k.id = c.culture_id WHERE c.id = ?", (constellation_id,)
    ).fetchone()
    namesets = _grouped_names(
        connection, "constellation_id = :id", {"id": constellation_id}, locale
    )
    return Constellation(
        id=constellation_id,
        culture_id=culture_id,
        iau=iau,
        names=namesets[0] if namesets else NameSet(culture_id, None, None, {}, None),
        attribution=attribution,
        prose=_prose(connection, constellation_id, culture_id, locale,
                     images_usable=bool(images_usable)) if with_prose else [],
    )


def _prose(connection: sqlite3.Connection, constellation_id: str, culture_id: str,
           locale: str, *, images_usable: bool) -> list[Prose]:
    """Both kinds of prose attached to a constellation, resolved per passage.

    Prose, so it resolves through `lang.prose_order` -- source first -- unlike the
    names beside it. Image references are stripped through the culture layer's rule
    even though no linked passage currently carries one: relying on "there are none
    today" is exactly the assumption that let `chart.webp` reach the article layer
    (TECHDEBT.md §5a), and the check costs nothing.
    """
    order = lang.prose_order(locale, lang.available_langs(connection))
    excluded = cultures.excluded_paths(connection, culture_id)

    def clean(text: str) -> str:
        stripped, _ = cultures.strip_unservable_images(
            text, images_usable=images_usable, excluded=excluded)
        return stripped

    passages: list[Prose] = []

    by_lang = {
        row[0]: row[1] for row in connection.execute(
            "SELECT lang, text FROM constellation_descriptions WHERE constellation_id = ?",
            (constellation_id,),
        )
    }
    note = lang.pick(by_lang, order, locale)
    if note:
        passages.append(Prose(text=clean(note.value), lang=note.lang, source="index",
                              heading_path=None, section_ord=None))

    sections: dict[int, dict[str, tuple[str, str]]] = {}
    for ordinal, section_lang, heading_path, text in connection.execute(
        "SELECT ord, lang, heading_path, text FROM sections WHERE constellation_id = ?",
        (constellation_id,),
    ):
        sections.setdefault(ordinal, {})[section_lang] = (heading_path, text)

    for ordinal in sorted(sections):
        variants = sections[ordinal]
        resolved = lang.pick({k: v[1] for k, v in variants.items()}, order, locale)
        if resolved:
            passages.append(Prose(
                text=clean(resolved.value),
                lang=resolved.lang,
                source="article",
                heading_path=variants[resolved.lang][0],
                section_ord=ordinal,
            ))
    return passages
