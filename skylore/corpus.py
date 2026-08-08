"""Readers for the three upstream formats: description.md, index.json, common_names.tab.

Parsing only -- nothing here touches the database.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import po

# Top-level `##` headings the corpus uses. Anything else is kept as "other" rather
# than dropped, so an upstream addition shows up in the data instead of vanishing.
KNOWN_KINDS = {
    "introduction", "description", "extras", "constellations",
    "references", "authors", "license",
}

# Boilerplate: needed for attribution, useless as retrieval targets, and actively
# harmful there because every culture's copy looks alike to a similarity search.
NON_RETRIEVABLE_KINDS = {"authors", "license"}

_HEADING = re.compile(r"^(#{2,6})[ \t]+(.+?)[ \t]*$", re.M)
_REFERENCE = re.compile(r"^[-*\s]*\[#(\d+)\]:\s*(.+)$", re.M)


@dataclass
class Subsection:
    """One retrieval unit: a heading and the prose under it."""
    level: int
    heading: str | None  # None for the lead text that precedes a section's first subheading
    body: str
    # Set when this subsection had no translation and holds another language's text
    # inside an otherwise translated article. See `localise_section`.
    fallback_from: str | None = None


@dataclass
class TopSection:
    kind: str
    heading: str
    body: str
    subsections: list[Subsection] = field(default_factory=list)


@dataclass
class Constellation:
    id: str
    ord: int
    gloss: str | None       # `common_name.english`: the meaning, not a language tag
    native: str | None
    pronounce: str | None
    iau: str | None
    description: str | None
    image_path: str | None
    image_size: tuple[int, int] | None
    anchors: list[tuple[int, int, int]]      # (x, y, hip)
    lines: list[tuple[str | None, list[int]]]  # (line weight, polyline of HIP numbers)


@dataclass
class CultureIndex:
    id: str
    region: str | None
    classification: list[str]
    native_lang: str | None
    highlight: str | None
    constellations: list[Constellation]
    star_names: dict[int, list[dict]]        # hip -> name records, most important first


# ───────────────────────────────── description.md ─────────────────────────────────

def split_headings(text: str, *, minimum_level: int = 3) -> list[Subsection]:
    """Break a section body at headings of `minimum_level` or deeper."""
    matches = [m for m in _HEADING.finditer(text) if len(m.group(1)) >= minimum_level]
    if not matches:
        return [Subsection(level=minimum_level - 1, heading=None, body=text.strip())]

    subsections: list[Subsection] = []
    lead = text[: matches[0].start()].strip()
    if lead:
        subsections.append(Subsection(level=minimum_level - 1, heading=None, body=lead))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        subsections.append(Subsection(
            level=len(match.group(1)),
            heading=match.group(2).strip(),
            body=text[match.end():end].strip(),
        ))
    return subsections


def read_title(markdown: str) -> str | None:
    """The `# <name>` heading: the culture's display name, and the msgid its
    translations are keyed on."""
    match = re.search(r"^#[ \t]+(.+?)[ \t]*$", markdown, re.M)
    return match.group(1).strip() if match else None


def read_article(markdown: str) -> list[TopSection]:
    """Split `description.md` into its `##` sections, each split again at `###`+."""
    sections: list[TopSection] = []
    matches = [m for m in _HEADING.finditer(markdown) if len(m.group(1)) == 2]
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        heading = match.group(2).strip()
        body = markdown[match.end():end].strip()
        slug = re.sub(r"[^a-z]+", "", heading.lower())
        sections.append(TopSection(
            kind=slug if slug in KNOWN_KINDS else "other",
            heading=heading,
            body=body,
            subsections=split_headings(body),
        ))
    return sections


def parse_references(body: str) -> dict[int, str]:
    """`- [#3]: Allen, R. H. …` -> {3: "Allen, R. H. …"}"""
    return {int(num): text.strip() for num, text in _REFERENCE.findall(body)}


def cited_refs(text: str) -> set[int]:
    """Which `[#N]` markers a passage actually uses."""
    return {int(n) for n in re.findall(r"\[#(\d+)\]", text)}


def summarise(introduction: str, limit: int = 400) -> str:
    """Opening sentences of an Introduction, for the culture catalogue.

    Deterministic on purpose: a generated summary would need regenerating per model
    and could not be diffed against the source.
    """
    plain = re.sub(r"\s+", " ", re.sub(r"!\[[^\]]*\]\([^)]*\)", "", introduction)).strip()
    if len(plain) <= limit:
        return plain
    cut = plain[:limit]
    stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[: stop + 1] if stop > limit // 2 else cut.rstrip() + "…").strip()


# ──────────────────────────────────── index.json ────────────────────────────────────

# `tukano/index.json` spells the field `descritpion`. Normalising known aliases keeps
# the typo from silently costing us that culture's per-constellation prose.
FIELD_ALIASES = {"descritpion": "description"}


def _normalise_fields(record: dict) -> dict:
    return {FIELD_ALIASES.get(key, key): value for key, value in record.items()}


def _parse_segment(segment: list) -> tuple[str | None, list[int]]:
    """Split a polyline into its optional line-weight hint and its HIP numbers.

    `western` and `western_SnT` prefix some segments with "thin" or "bold" to vary the
    stroke; every other culture gives bare HIP numbers.
    """
    styles = [v for v in segment if isinstance(v, str)]
    return styles[0] if styles else None, [v for v in segment if isinstance(v, int)]


def read_index(path: Path) -> CultureIndex:
    raw = json.loads(path.read_text(encoding="utf-8"))

    constellations: list[Constellation] = []
    for ordinal, entry in enumerate(raw.get("constellations", [])):
        entry = _normalise_fields(entry)
        name = entry.get("common_name", {})
        image = entry.get("image") or {}
        size = image.get("size")
        constellations.append(Constellation(
            id=entry["id"],
            ord=ordinal,
            gloss=name.get("english"),
            native=name.get("native"),
            pronounce=name.get("pronounce"),
            iau=entry.get("iau"),
            description=entry.get("description"),
            image_path=image.get("file"),
            image_size=(size[0], size[1]) if size else None,
            anchors=[(a["pos"][0], a["pos"][1], a["hip"]) for a in image.get("anchors", [])],
            lines=[_parse_segment(segment) for segment in entry.get("lines", [])],
        ))

    star_names: dict[int, list[dict]] = {}
    for key, records in raw.get("common_names", {}).items():
        match = re.fullmatch(r"HIP\s+(\d+)", key.strip())
        if match:  # a few keys use other catalogues; HIP is the only one we can join on
            star_names[int(match.group(1))] = [_normalise_fields(r) for r in records]

    return CultureIndex(
        id=raw["id"],
        region=raw.get("region"),
        classification=raw.get("classification", []),
        native_lang=raw.get("native_lang"),
        highlight=raw.get("highlight"),
        constellations=constellations,
        star_names=star_names,
    )


# ─────────────────────────────── common_names.tab ───────────────────────────────

@dataclass
class InternationalName:
    hip: int
    name: str
    designation: str | None
    rank: int


def read_common_names(path: Path) -> list[InternationalName]:
    """International star names, in the source's order of importance per star.

    Only `model == "star"` rows: the file also keys deep-sky objects by HIP, which
    would otherwise be mistaken for stars.
    """
    seen: dict[int, int] = {}
    names: list[InternationalName] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        columns = [c.strip() for c in line.split("|")]
        if len(columns) < 4 or columns[3] != "star":
            continue
        match = re.fullmatch(r"HIP\s+(\d+)", columns[0])
        if not match:
            continue
        hip = int(match.group(1))
        rank = seen.get(hip, 0)
        seen[hip] = rank + 1
        note = columns[5] if len(columns) > 5 else ""
        bayer = re.fullmatch(r"Star\s+(.+)", note.strip())
        names.append(InternationalName(
            hip=hip,
            name=columns[1],
            designation=bayer.group(1).strip() if bayer else None,
            rank=rank,
        ))
    return names


# ─────────────────────────────────── localisation ───────────────────────────────────

def localise_section(
    section: TopSection,
    table: dict[str, dict[str, str]],
    lang: str,
) -> tuple[list[Subsection], str | None]:
    """Translated subsections of one `##` section, plus a warning if any.

    Two upstream shapes have to be handled. Most sections are translated as a single
    catalogue entry covering the whole `##` body, so the translation must be re-split
    at its headings and aligned with the English subsections positionally. The
    `## Constellations` section has no whole-body entry at all -- upstream extracts
    each `##### <native name>` subsection separately -- so those are looked up one by
    one, and untranslated ones fall back to English individually.

    Returns ([], None) when nothing in this section is available in `lang`.
    """
    whole = table.get(po.normalise(section.body), {}).get(lang)
    if whole:
        translated = split_headings(whole)
        if len(translated) == len(section.subsections):
            return translated, None
        # Heading structure diverged from the source. Keep the text rather than risk
        # attaching prose to the wrong heading.
        return (
            [Subsection(level=2, heading=None, body=whole.strip())],
            f"{section.kind}: {len(section.subsections)} subsections in English, "
            f"{len(translated)} in {lang}; stored unsplit",
        )

    per_subsection: list[Subsection] = []
    any_translated = False
    for subsection in section.subsections:
        body = table.get(po.normalise(subsection.body), {}).get(lang)
        any_translated |= body is not None
        per_subsection.append(Subsection(
            level=subsection.level,
            heading=subsection.heading,   # native-name headings are not translated
            body=(body or subsection.body).strip(),
            fallback_from=None if body else "en",
        ))
    return (per_subsection, None) if any_translated else ([], None)
