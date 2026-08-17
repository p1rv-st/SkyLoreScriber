"""Minimal reader for the gettext catalogues shipped with each sky culture.

Only what this corpus actually uses: singular entries, extracted comments, fuzzy
flags. No plurals, no contexts, no writing. `polib` would do the same job, but the
subset is small enough that a dependency is not worth the supply chain.

The important detail is how translations are keyed. Upstream builds these catalogues
from `description.md` and `index.json`, so every `msgid` *is* the English source
string -- a whole markdown section, a constellation gloss, a star name. Matching
translations back to their source by msgid is therefore exact, and does not depend on
the wording of the `#.` comment, which varies per culture.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

# A .po string is one or more adjacent quoted chunks, each C-escaped.
_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Entry:
    comment: str  # the `#.` extracted comment, "" when absent
    msgid: str
    msgstr: str
    fuzzy: bool

    @property
    def translated(self) -> bool:
        return bool(self.msgstr) and not self.fuzzy


def _unquote(block: str, keyword: str) -> str | None:
    """Join and unescape the quoted chunks following `keyword` in one entry block."""
    match = re.search(rf'^{keyword}((?:[ \t]*"(?:[^"\\]|\\.)*"\s*)+)', block, re.M)
    if match is None:
        return None
    # ast.literal_eval handles the C escapes (\n, \", \\) with the same rules gettext uses.
    return "".join(ast.literal_eval(f'"{chunk}"') for chunk in _STRING.findall(match.group(1)))


def read(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    for block in re.split(r"\n\n+", path.read_text(encoding="utf-8")):
        if "Project-Id-Version" in block:  # the header pseudo-entry
            continue
        msgid = _unquote(block, "msgid")
        if msgid is None or not msgid:
            continue
        comment = re.search(r"^#\.\s*(.+)$", block, re.M)
        entries.append(Entry(
            comment=comment.group(1).strip() if comment else "",
            msgid=msgid,
            msgstr=_unquote(block, "msgstr") or "",
            fuzzy=bool(re.search(r"^#,.*\bfuzzy\b", block, re.M)),
        ))
    return entries


def normalise(source: str) -> str:
    """Key for matching a source string to its msgid.

    Line wrapping differs between `description.md` and the catalogues, so collapse
    whitespace. Everything else is compared verbatim.
    """
    return re.sub(r"\s+", " ", source).strip()


def translations(culture_dir: Path, languages: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """{normalised English source: {language: translation}} for one culture.

    English is absent by design: `en.po` carries empty msgstrs upstream, because
    `description.md` *is* the English text.
    """
    table: dict[str, dict[str, str]] = {}
    for lang in languages:
        po_path = culture_dir / "po" / f"{lang}.po"
        if not po_path.exists():
            continue
        for entry in read(po_path):
            if entry.translated:
                table.setdefault(normalise(entry.msgid), {})[lang] = entry.msgstr
    return table
