"""The agent-facing boundary: four tools, their schemas, and JSON serialisation.

Everything below this module returns frozen dataclasses. Everything above it speaks
JSON. That translation is not mechanical, and `dataclasses.asdict` would be the wrong
answer three times over:

**Provenance has to survive.** `lang.Resolved` carries the language a value actually
came from. Flattening it to a bare string drops the "this is a fallback, not a
translation" signal at the last step -- after the resolver was built specifically to
produce it. Russian star-gloss coverage is 4%, so that signal fires constantly.

**Attribution belongs to the response, not to every object in it.** Aldebaran is drawn
by 27 cultures; 27 copies of their licence texts would be most of the payload. The
`sources` block carries each culture once, and `_with_sources` derives it from the
payload rather than trusting a caller to remember -- the licensing invariant is a
property of what reaches the model, so this is the layer that has to hold it.

**Native names are not text to be rendered.** `native` and `pronounce` are kept in
their own fields with no language attached, so a model reading the JSON has no reason
to treat 毕宿 as English awaiting translation.

`names.search` is deliberately not exposed. It is mechanism, and a tool named after
mechanism makes the model decide "do I want exact or fuzzy matching here" -- an
information-retrieval question models answer badly (PLAN.md §3).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from . import compare, cultures, lang, names, retrieval
from .paths import DATABASE


def connect(path: Path | str = DATABASE) -> sqlite3.Connection:
    """A read-only connection. Nothing in the query layer writes, so nothing may."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


# ───────────────────────────────── serialisation ─────────────────────────────────

def _resolved(value: lang.Resolved | None) -> dict[str, Any] | None:
    """A resolved value with its provenance, flagged loudly when it is a fallback.

    `is_fallback` is present only when true. An exception that is always spelled out
    is easy to skim past; one that appears only when it applies is not.
    """
    if value is None:
        return None
    payload: dict[str, Any] = {"value": value.value, "lang": value.lang}
    if not value.matches_request:
        payload["is_fallback"] = True
        payload["requested"] = value.requested
    return payload


def _nameset(nameset: names.NameSet) -> dict[str, Any]:
    """Every name one culture has for an object.

    The whole dictionary, not the preferred string alone: a single resolved name
    would be English wearing a Russian label in almost every star lookup, and the
    model could not tell. `native` and `pronounce` carry no language by schema CHECK
    and keep their own fields here so they are not mistaken for translatable text.
    """
    payload: dict[str, Any] = {"culture": nameset.culture_id}
    if nameset.native:
        payload["native"] = nameset.native
    if nameset.pronounce:
        payload["pronounce"] = nameset.pronounce
    if nameset.glosses:
        payload["meanings"] = nameset.glosses
    preferred = _resolved(nameset.preferred)
    if preferred:
        payload["preferred"] = preferred
    elif nameset.native:
        # No gloss resolved, so the name to use in prose is the untranslated one.
        payload["preferred"] = {"value": nameset.native, "lang": None}
    return payload


def _prose(passage: names.Prose) -> dict[str, Any]:
    payload = {"text": passage.text, "lang": passage.lang, "source": passage.source}
    if passage.heading_path:
        payload["heading_path"] = passage.heading_path
    return payload


def _constellation(constellation: names.Constellation) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": constellation.id,
        "culture": constellation.culture_id,
        "names": _nameset(constellation.names),
    }
    if constellation.iau:
        payload["iau"] = constellation.iau
    if constellation.prose:
        payload["prose"] = [_prose(p) for p in constellation.prose]
    return payload


def _star(star: names.Star) -> dict[str, Any]:
    payload: dict[str, Any] = {"hip": star.hip}
    if star.iau_name:
        payload["international_name"] = star.iau_name
    if star.designation:
        payload["designation"] = star.designation
    payload["named_by"] = [_nameset(n) for n in star.names]
    # Naming and drawing are different relations: a culture can draw a figure through
    # a star it never names, and name a star it draws no figure through. Reporting
    # only one would understate every cross-culture question.
    payload["drawn_into"] = [_constellation(c) for c in star.figures]
    return payload


def _card(card: cultures.CultureCard) -> dict[str, Any]:
    return {
        "id": card.id,
        "name": _resolved(card.name),
        "region": card.region,
        "classification": card.classification,
        "summary": _resolved(card.summary),
        "constellation_count": card.constellation_count,
    }


def _section(section: cultures.Section) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "heading_path": section.heading_path,
        "kind": section.kind,
        "text": section.text,
        "lang": section.lang,
    }
    if section.references:
        # Resolved here rather than stored inline, so the database keeps the source
        # verbatim. Without this the model receives a bare "[#7]".
        payload["references"] = {str(n): t for n, t in section.references.items()}
    if section.fallback_from:
        payload["fallback_from"] = section.fallback_from
    if section.constellation_id:
        payload["constellation"] = section.constellation_id
    return payload


# ───────────────────────── licensing at the boundary ─────────────────────────

def _identifiers(payload: Any) -> set[str]:
    """Every string filed under an id-ish key, anywhere in a response."""
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in ("culture", "culture_id", "id") and isinstance(value, str):
                found.add(value)
            else:
                found |= _identifiers(value)
    elif isinstance(payload, list):
        for item in payload:
            found |= _identifiers(item)
    return found


def known_cultures(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT id FROM cultures")}


def _cultures_mentioned(connection: sqlite3.Connection, payload: Any) -> set[str]:
    """Every culture a response drew on, however deeply nested.

    `id` means different things at different depths -- a culture id on a catalogue
    card, a constellation id like "CON western Tau" on a figure -- so candidates are
    checked against the culture table rather than trusted by key name. Over-collecting
    would not leak anything (an unknown id simply matches no row), but it would make
    this function useless for *asserting* the invariant, which is its main job.
    """
    return _identifiers(payload) & known_cultures(connection)


def _with_sources(connection: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    """Attach attribution and licence terms for every culture the response used.

    Derived from the payload, not passed in. `cultures.attribution` travelling with
    every answer is a licence condition, and a condition a caller has to remember is
    one that eventually gets forgotten.

    Artwork terms are carried alongside the prose terms because they differ: Free Art
    License is copyleft, so anything showing those images owes LAL terms rather than
    only the prose CC-BY-SA.
    """
    ids = _cultures_mentioned(connection, payload)
    if not ids:
        return payload
    placeholders = ",".join("?" * len(ids))
    rows = connection.execute(
        f"SELECT id, attribution, text_licenses, image_licenses, images_usable"
        f"  FROM cultures WHERE id IN ({placeholders})", sorted(ids)
    ).fetchall()
    payload["sources"] = {
        culture_id: {
            "attribution": attribution,
            "text_licenses": json.loads(text_licenses or "[]"),
            "image_licenses": json.loads(image_licenses or "[]"),
            "images_usable": bool(images_usable),
        }
        for culture_id, attribution, text_licenses, image_licenses, images_usable in rows
    }
    return payload


# ───────────────────────────────────── tools ─────────────────────────────────────

def find_cultures(connection: sqlite3.Connection, *, query: str | None = None,
                  region: str | None = None, lang: str = "en") -> dict[str, Any]:
    cards = cultures.find_cultures(connection, query=query, region=region, locale=lang)
    return _with_sources(connection, {"cultures": [_card(c) for c in cards]})


def get_culture_article(connection: sqlite3.Connection, *, culture: str,
                        lang: str = "en", section: str | None = None) -> dict[str, Any]:
    article = cultures.get_culture_article(connection, culture, locale=lang,
                                           section=section)
    if article is None:
        return {"error": f"no such culture: {culture!r}",
                "hint": "call find_cultures to list the 34 available ids"}
    payload: dict[str, Any] = {
        "culture": article.culture_id,
        "name": _resolved(article.name),
        "sections": [_section(s) for s in article.sections],
    }
    if article.omitted_images:
        # Counted, not named. The model gains something real from knowing an image was
        # removed -- japanese_moon_stations otherwise shows a caption with nothing
        # above it, and a model that does not know why will invent the missing figure.
        # It gains nothing from the filename, and handing over the path of an asset we
        # are forbidden to serve invites it back into the answer as a citation. The
        # paths stay on `Article.omitted_images` for whoever is debugging.
        payload["omitted"] = {
            "images": len(article.omitted_images),
            "reason": "removed by licence; do not refer to or request these figures",
        }
    return _with_sources(connection, payload)


def lookup_star(connection: sqlite3.Connection, *, query: str, lang: str = "en",
                limit: int = 10) -> dict[str, Any]:
    found = names.lookup_star(connection, query, locale=lang, limit=limit)
    return _with_sources(connection, {"stars": [_star(s) for s in found]})


def find_constellation(connection: sqlite3.Connection, *, query: str,
                       lang: str = "en", culture: str | None = None,
                       limit: int = 10) -> dict[str, Any]:
    found = names.find_constellation(connection, query, locale=lang, culture=culture,
                                     limit=limit)
    return _with_sources(
        connection, {"constellations": [_constellation(c) for c in found]})


def _passage(passage: retrieval.Passage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "culture": passage.culture_id,
        "heading_path": passage.heading_path,
        "kind": passage.kind,
        "text": passage.text,
        "lang": passage.lang,
    }
    if passage.cross_language:
        # The query matched one language and the text is served in another. Said out
        # loud because the model is about to translate this for the user, and it should
        # know the match was not made on the words it is reading.
        payload["matched_lang"] = passage.matched_lang
    if passage.references:
        payload["references"] = {str(n): t for n, t in passage.references.items()}
    return payload


def search_lore(connection: sqlite3.Connection, *, query: str, lang: str = "en",
                culture: str | None = None, limit: int = retrieval.TOP_K,
                embedder: retrieval.Embedder | None = None) -> dict[str, Any]:
    found = retrieval.search_lore(connection, query, locale=lang, culture=culture,
                                  limit=limit, embedder=embedder)
    return _with_sources(connection, {"passages": [_passage(p) for p in found]})


def _overlap(overlap: compare.Overlap) -> dict[str, Any]:
    return {
        "culture": overlap.culture_id,
        "constellation": overlap.constellation_id,
        "names": _nameset(overlap.names),
        "shared_stars": overlap.shared,
        "shared_hips": overlap.shared_hips,
        # Both fractions, because one number cannot distinguish "covers most of the sky
        # you asked about" from "is entirely explained by it", and the difference is the
        # whole point: the Belarusian Throne of Jesus lies wholly inside Orion while the
        # Egyptian Sah merely crosses it.
        "share_of_asked": round(overlap.of_target, 2),
        "share_of_figure": round(overlap.of_figure, 2),
    }


def compare_across_cultures(connection: sqlite3.Connection, *,
                            constellation: str | None = None,
                            hips: list[int] | None = None,
                            lang: str = "en", limit: int = 20) -> dict[str, Any]:
    if not constellation and not hips:
        return {"error": "give either `constellation` or `hips`",
                "hint": "find_constellation returns constellation ids; lookup_star "
                        "returns HIP numbers"}
    result = compare.compare_across_cultures(
        connection, constellation_id=constellation, hips=hips, locale=lang, limit=limit)
    if not result.target_hips:
        return {"error": f"no star pattern found for {constellation or hips!r}",
                "hint": "some constellations carry names but no line figure"}
    payload: dict[str, Any] = {
        "asked_about": {"hips": result.target_hips, "star_count": len(result.target_hips)},
        "seen_elsewhere_as": [_overlap(o) for o in result.overlaps],
    }
    if result.target:
        payload["asked_about"]["constellation"] = _constellation(result.target)
    return _with_sources(connection, payload)


TOOLS = {
    "find_cultures": find_cultures,
    "compare_across_cultures": compare_across_cultures,
    "get_culture_article": get_culture_article,
    "lookup_star": lookup_star,
    "find_constellation": find_constellation,
    "search_lore": search_lore,
}


def call(connection: sqlite3.Connection, name: str, arguments: dict) -> dict[str, Any]:
    """Dispatch one tool call. Unknown names and bad arguments answer, never raise:
    a tool that throws teaches the model nothing about what to try instead."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"no such tool: {name!r}", "available": sorted(TOOLS)}
    try:
        return tool(connection, **arguments)
    except TypeError as error:
        return {"error": f"bad arguments for {name}: {error}"}


# ───────────────────────────────── tool schemas ─────────────────────────────────

_LANG = {
    "type": "string",
    "description": (
        "BCP 47 tag for the user's language, e.g. 'ru', 'es', 'zh-Hans'. Names come "
        "back in this language where the corpus has them; prose comes back in "
        "English, which is the language it was written in -- translate it yourself "
        "rather than asking for a translated article. Defaults to 'en'."
    ),
}

SCHEMAS = [
    {
        "name": "find_cultures",
        "description": (
            "List the sky cultures in the corpus: 34 of them, with a one-line summary, "
            "region and constellation count each. Returns the whole catalogue, so read "
            "it and choose -- this is a directory, not a search. Start here when the "
            "question names no specific culture, star or constellation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description":
                          "Optional substring filter over names, summaries and ids, in "
                          "any language. Narrows the list; it does not rank it."},
                "region": {"type": "string", "enum":
                           ["Europe", "Asia", "America", "Oceania", "Middle East"]},
                "lang": _LANG,
            },
        },
    },
    {
        "name": "get_culture_article",
        "description": (
            "Read one culture's article in full. Articles average ~5700 characters and "
            "are returned whole and unchunked, so prefer reading one completely over "
            "sampling several. Citation markers like [#3] are resolved to their sources "
            "in the same response. Use `section` for lokono, which is ten times the "
            "average length."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "culture": {"type": "string", "description":
                            "Culture id from find_cultures, e.g. 'lokono', 'chinese'."},
                "section": {"type": "string", "description":
                            "Optional. A section kind ('introduction', 'description', "
                            "'constellations', 'extras', 'references') or a heading "
                            "path prefix such as 'Description'."},
                "lang": _LANG,
            },
            "required": ["culture"],
        },
    },
    {
        "name": "lookup_star",
        "description": (
            "Find a star and everything the corpus knows about it: which cultures name "
            "it, what each calls it, and which constellation figures are drawn through "
            "it. Accepts any key the user might have -- an HIP number ('HIP 21421' or "
            "'21421'), an international name ('Aldebaran'), a Bayer designation "
            "('alf Tau'), a native name from any culture ('Wara-wara'), or a meaning in "
            "any of the four languages. Naming and drawing are reported separately: a "
            "culture can draw a figure through a star it never names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "Default 10."},
                "lang": _LANG,
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_constellation",
        "description": (
            "Find a constellation by name across every culture and language, with fuzzy "
            "matching. Returns each match with all its names -- the native name, its "
            "romanisation, and the meaning in each language -- plus any prose the "
            "corpus has about it. Search in the user's own language: the corpus holds "
            "Chinese, Russian and Spanish names directly, so there is no need to "
            "translate the query first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "culture": {"type": "string", "description":
                            "Optional culture id to search within."},
                "limit": {"type": "integer", "description": "Default 10."},
                "lang": _LANG,
            },
            "required": ["query"],
        },
    },
    {
        "name": "compare_across_cultures",
        "description": (
            "Given one constellation, or a list of HIP numbers, find every other "
            "tradition that draws a figure through the same stars and report what each "
            "one saw there — western Orion comes back as the Tupi Old Man, the Navajo "
            "First Slim One, the Egyptian Sah. This is the cross-cultural comparison "
            "tool; use it whenever the question is 'who else', 'how do other cultures "
            "see', or 'what does this look like elsewhere'. Results carry two shares: "
            "`share_of_asked` is how much of the sky you asked about the figure covers, "
            "`share_of_figure` is how much of that figure your stars explain. A small "
            "figure lying wholly inside a large one has a low first and a full second."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "constellation": {"type": "string", "description":
                                  "Constellation id from find_constellation, e.g. "
                                  "'CON western Ori'."},
                "hips": {"type": "array", "items": {"type": "integer"},
                         "description": "HIP numbers to compare instead of a "
                                        "constellation, e.g. the Pleiades."},
                "limit": {"type": "integer", "description": "Default 20."},
                "lang": _LANG,
            },
        },
    },
    {
        "name": "search_lore",
        "description": (
            "Search the prose by meaning, for questions that name no star, constellation "
            "or culture -- 'which cultures tie a star to the rainy season', 'how were "
            "eclipses explained'. Returns whole sections, each with the culture and "
            "heading path it came from, so an answer can be attributed. Ask in the "
            "user's own language: all four are indexed. Use the other tools when the "
            "question does name something; this one is for the cases they cannot reach."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description":
                          "The information need, phrased as a statement or question. "
                          "Full sentences work better than keywords."},
                "culture": {"type": "string", "description":
                            "Optional culture id to search within. Without it results "
                            "are spread across cultures; with it you get everything "
                            "that culture says on the topic."},
                "limit": {"type": "integer", "description": "Default 8."},
                "lang": _LANG,
            },
            "required": ["query"],
        },
    },
]


# ────────────────────────────────────── cli ──────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Call a tool from the shell and print what the model would see.

        python -m skylore.tools find_constellation '{"query": "毕宿", "lang": "zh-Hans"}'
        python -m skylore.tools --schemas
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("tool", nargs="?", help="tool name")
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON object")
    parser.add_argument("--schemas", action="store_true",
                        help="print the tool definitions the model receives")
    args = parser.parse_args(argv)

    if args.schemas:
        print(json.dumps(SCHEMAS, indent=2, ensure_ascii=False))
        return 0
    if not args.tool:
        parser.error("a tool name is required (or --schemas)")

    with connect() as connection:
        result = call(connection, args.tool, json.loads(args.arguments))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
