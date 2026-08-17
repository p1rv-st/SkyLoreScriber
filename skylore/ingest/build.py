"""Build corpus.db from the sky culture submodule.

    python -m skylore.ingest [--force] [--culture ID ...]

Only cultures on the allowlist are ingested; the rest are recorded in
`skipped_cultures` with the licence reasons, so an absence is never ambiguous.

Re-runs are incremental: a culture whose `description.md` and `index.json` hash to
the same value as last time is left alone. `--force` rebuilds everything.

Embeddings are not written here -- that needs a model loaded and belongs in its own
step. Everything the structured tools need (`lookup_star`, `find_constellation`,
`compare_across_cultures`, `get_culture_article`) is complete after this runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..lang import LANGUAGES, SOURCE_LANG, TRANSLATION_LANGS  # noqa: F401  (re-exported)
from ..paths import ALLOWLIST, CORPUS_DIR, DATABASE, SCHEMA
from . import corpus, po

SCHEMA_VERSION = 3  # 2: names_fts folds diacritics; 3: embeddings keyed by (section, model)


def corpus_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(CORPUS_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def source_hash(culture_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("description.md", "index.json"):
        path = culture_dir / name
        if path.exists():
            digest.update(path.read_bytes())
    for po_path in sorted((culture_dir / "po").glob("*.po")):
        if po_path.stem in LANGUAGES:
            digest.update(po_path.read_bytes())
    return digest.hexdigest()


def connect(force: bool) -> sqlite3.Connection:
    if force and DATABASE.exists():
        DATABASE.unlink()
    fresh = not DATABASE.exists()
    connection = sqlite3.connect(DATABASE)
    connection.executescript(SCHEMA.read_text(encoding="utf-8")) if fresh else None
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


class Ingest:
    def __init__(self, connection: sqlite3.Connection):
        self.db = connection
        self.warnings: list[str] = []
        # Every HIP the corpus mentions, so lines and names always have a referent.
        self._known_stars: set[int] = {
            row[0] for row in self.db.execute("SELECT hip FROM stars")
        }

    # ── stars ──────────────────────────────────────────────────────────────────

    def register_stars(self, hips: set[int]) -> None:
        new = hips - self._known_stars
        if not new:
            return
        self.db.executemany("INSERT INTO stars(hip) VALUES (?)", [(h,) for h in sorted(new)])
        self._known_stars |= new

    def load_international_names(self, path: Path) -> int:
        names = corpus.read_common_names(path)
        self.register_stars({n.hip for n in names})
        for name in names:
            if name.rank == 0:
                self.db.execute(
                    "UPDATE stars SET iau_name = ?, designation = ?, named_by = ? WHERE hip = ?",
                    (name.name, name.designation, "common_names.tab", name.hip),
                )
            self.db.execute(
                "INSERT INTO names(hip, culture_id, kind, lang, value, rank) "
                "VALUES (?, NULL, 'gloss', ?, ?, ?)",
                (name.hip, SOURCE_LANG, name.name, name.rank),
            )
        return len(names)

    # ── one culture ────────────────────────────────────────────────────────────

    def culture(self, entry: dict, culture_dir: Path) -> None:
        culture_id = entry["id"]
        index = corpus.read_index(culture_dir / "index.json")
        markdown = (culture_dir / "description.md").read_text(encoding="utf-8")
        article = corpus.read_article(markdown)
        table = po.translations(culture_dir, TRANSLATION_LANGS)

        authors = next((s.body for s in article if s.kind == "authors"), "")
        images = entry.get("images", {})
        self.db.execute(
            "INSERT INTO cultures(id, region, classification, native_lang, highlight,"
            " constellation_count, text_licenses, text_commercial, text_share_alike,"
            " image_licenses, images_usable, attribution, source_sha256)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                culture_id, index.region, json.dumps(index.classification),
                index.native_lang, index.highlight, len(index.constellations),
                json.dumps(entry["licenses"]), int(entry["commercial"]),
                int(entry["share_alike"]), json.dumps(images.get("licenses", [])),
                int(images.get("usable", False)), authors, source_hash(culture_dir),
            ),
        )
        for path in images.get("excluded", []):
            self.db.execute(
                "INSERT OR IGNORE INTO excluded_assets(culture_id, path, reason) VALUES (?,?,?)",
                (culture_id, path, entry.get("manual_review", {}).get("rationale", "manual review")),
            )

        self._constellations(culture_id, index, table)
        self._star_names(culture_id, index, table)
        self._sections(culture_id, article, table)
        self._culture_names(culture_id, corpus.read_title(markdown), article, table)

    def _constellations(self, culture_id: str, index: corpus.CultureIndex,
                        table: dict[str, dict[str, str]]) -> None:
        hips = {hip for c in index.constellations for _, line in c.lines for hip in line}
        hips |= {a[2] for c in index.constellations for a in c.anchors}
        self.register_stars(hips)

        for constellation in index.constellations:
            self.db.execute(
                "INSERT INTO constellations(id, culture_id, ord, iau, image_path, image_w, image_h)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    constellation.id, culture_id, constellation.ord, constellation.iau,
                    constellation.image_path,
                    constellation.image_size[0] if constellation.image_size else None,
                    constellation.image_size[1] if constellation.image_size else None,
                ),
            )
            self.db.executemany(
                "INSERT INTO constellation_lines(constellation_id, segment, seq, hip, style)"
                " VALUES (?,?,?,?,?)",
                [(constellation.id, segment, seq, hip, style if seq == 0 else None)
                 for segment, (style, line) in enumerate(constellation.lines)
                 for seq, hip in enumerate(line)],
            )
            self.db.executemany(
                "INSERT INTO image_anchors(constellation_id, idx, x, y, hip) VALUES (?,?,?,?,?)",
                [(constellation.id, i, x, y, hip)
                 for i, (x, y, hip) in enumerate(constellation.anchors)],
            )

            # A native name and its romanisation are the same in every interface
            # language; only the gloss is translated.
            for kind, value in (("native", constellation.native),
                                ("pronounce", constellation.pronounce)):
                if value:
                    self.db.execute(
                        "INSERT INTO names(constellation_id, culture_id, kind, lang, value)"
                        " VALUES (?,?,?,NULL,?)",
                        (constellation.id, culture_id, kind, value),
                    )
            if constellation.gloss:
                self._glosses(
                    "constellation_id", constellation.id, culture_id,
                    constellation.gloss, table,
                )
            if constellation.description:
                self._descriptions(constellation.id, constellation.description, table)

    def _glosses(self, column: str, subject: str, culture_id: str, english: str,
                 table: dict[str, dict[str, str]], rank: int = 0) -> None:
        translated = table.get(po.normalise(english), {})
        for lang, value in [(SOURCE_LANG, english), *translated.items()]:
            self.db.execute(
                f"INSERT INTO names({column}, culture_id, kind, lang, value, rank)"
                " VALUES (?,?,'gloss',?,?,?)",
                (subject, culture_id, lang, value, rank),
            )

    def _descriptions(self, constellation_id: str, english: str,
                      table: dict[str, dict[str, str]]) -> None:
        translated = table.get(po.normalise(english), {})
        self.db.executemany(
            "INSERT INTO constellation_descriptions(constellation_id, lang, text)"
            " VALUES (?,?,?)",
            [(constellation_id, lang, text)
             for lang, text in [(SOURCE_LANG, english), *translated.items()]],
        )

    def _star_names(self, culture_id: str, index: corpus.CultureIndex,
                    table: dict[str, dict[str, str]]) -> None:
        self.register_stars(set(index.star_names))
        for hip, records in index.star_names.items():
            for rank, record in enumerate(records):
                for kind in ("native", "pronounce"):
                    if record.get(kind):
                        self.db.execute(
                            "INSERT INTO names(hip, culture_id, kind, lang, value, rank)"
                            " VALUES (?,?,?,NULL,?,?)",
                            (hip, culture_id, kind, record[kind], rank),
                        )
                if record.get("english"):
                    self._glosses("hip", str(hip), culture_id, record["english"], table, rank)

    def _culture_names(self, culture_id: str, title: str | None,
                       article: list[corpus.TopSection],
                       table: dict[str, dict[str, str]]) -> None:
        # The display name is the `# Title` heading; upstream keys its translations
        # on that exact string.
        display = title or culture_id
        for lang, value in [(SOURCE_LANG, display), *table.get(po.normalise(display), {}).items()]:
            self.db.execute(
                "INSERT OR REPLACE INTO culture_names(culture_id, lang, value) VALUES (?,?,?)",
                (culture_id, lang, value),
            )

        introduction = next((s.body for s in article if s.kind == "introduction"), "")
        if not introduction:
            return
        localised = {SOURCE_LANG: introduction}
        localised.update(table.get(po.normalise(introduction), {}))
        for lang, text in localised.items():
            self.db.execute(
                "INSERT OR REPLACE INTO culture_summaries(culture_id, lang, summary) VALUES (?,?,?)",
                (culture_id, lang, corpus.summarise(text)),
            )

    def _sections(self, culture_id: str, article: list[corpus.TopSection],
                  table: dict[str, dict[str, str]]) -> None:
        # `ord` is assigned from the English structure and reused for every language,
        # so the same passage carries the same ordinal regardless of language.
        ordinals: dict[tuple[int, int], int] = {}
        counter = 0
        for top_index, section in enumerate(article):
            for sub_index in range(len(section.subsections)):
                ordinals[(top_index, sub_index)] = counter
                counter += 1

        # Which constellation a `##### <native name>` subsection is about.
        by_native = {
            value.casefold(): constellation_id
            for constellation_id, value in self.db.execute(
                "SELECT constellation_id, value FROM names"
                " WHERE culture_id = ? AND kind = 'native' AND constellation_id IS NOT NULL",
                (culture_id,),
            )
        }

        for lang in LANGUAGES:
            for top_index, section in enumerate(article):
                if lang == SOURCE_LANG:
                    subsections, warning = section.subsections, None
                else:
                    subsections, warning = corpus.localise_section(section, table, lang)
                if warning:
                    self.warnings.append(f"{culture_id}/{lang}: {warning}")
                if not subsections:
                    continue  # absent in this language; the query layer falls back

                if section.kind == "references":
                    self.db.executemany(
                        "INSERT OR REPLACE INTO section_refs(culture_id, lang, ref_num, text)"
                        " VALUES (?,?,?,?)",
                        [(culture_id, lang, num, text)
                         for num, text in corpus.parse_references(subsections[0].body).items()],
                    )

                for sub_index, subsection in enumerate(subsections):
                    ordinal = ordinals.get((top_index, sub_index), ordinals[(top_index, 0)])
                    path = " › ".join(p for p in (section.heading, subsection.heading) if p)
                    self.db.execute(
                        "INSERT INTO sections(culture_id, lang, ord, kind, level,"
                        " heading, heading_path, text, constellation_id, fallback_from, retrievable)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            culture_id, lang, ordinal, section.kind, subsection.level,
                            subsection.heading, path, subsection.body,
                            by_native.get((subsection.heading or "").casefold()),
                            subsection.fallback_from,
                            # A heading whose body is empty -- because a deeper
                            # heading follows immediately -- is a structural node,
                            # not something worth retrieving.
                            int(section.kind not in corpus.NON_RETRIEVABLE_KINDS
                                and bool(subsection.body.strip())),
                        ),
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild the database from scratch")
    parser.add_argument("--culture", nargs="*", help="restrict to these culture ids")
    args = parser.parse_args()

    if not CORPUS_DIR.is_dir():
        print(f"missing corpus: {CORPUS_DIR}", file=sys.stderr)
        return 1
    if not ALLOWLIST.exists():
        print(f"missing {ALLOWLIST} -- run python -m scripts.scan_licenses first",
              file=sys.stderr)
        return 1

    allowlist = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    allowed = {e["id"]: e for e in allowlist["allowed"]}
    if args.culture:
        unknown = set(args.culture) - set(allowed)
        if unknown:
            print(f"not on the allowlist: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1
        allowed = {k: v for k, v in allowed.items() if k in args.culture}

    connection = connect(force=args.force)
    ingest = Ingest(connection)

    with connection:
        for rejected in allowlist["rejected"]:
            connection.execute(
                "INSERT OR REPLACE INTO skipped_cultures(id, reasons) VALUES (?,?)",
                (rejected["id"], json.dumps(rejected["reasons"])),
            )

        current = dict(connection.execute("SELECT id, source_sha256 FROM cultures"))
        done = skipped = 0
        for culture_id, entry in sorted(allowed.items()):
            culture_dir = CORPUS_DIR / culture_id
            if current.get(culture_id) == source_hash(culture_dir):
                skipped += 1
                continue
            # Cascades through constellations, names and sections.
            connection.execute("DELETE FROM cultures WHERE id = ?", (culture_id,))
            ingest.culture(entry, culture_dir)
            done += 1

        international = 0
        if done or args.force:
            connection.execute("DELETE FROM names WHERE culture_id IS NULL")
            international = ingest.load_international_names(CORPUS_DIR / "common_names.tab")

        for key, value in {
            "schema_version": str(SCHEMA_VERSION),
            "corpus_commit": corpus_commit(),
            "languages": json.dumps(list(LANGUAGES)),
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }.items():
            connection.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value))

        connection.execute("INSERT INTO names_fts(names_fts) VALUES ('rebuild')")
        connection.execute("INSERT INTO sections_fts(sections_fts) VALUES ('rebuild')")

    counts = {
        table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("cultures", "constellations", "stars", "names", "sections")
    }
    connection.close()

    print(f"{DATABASE.name}: {done} cultures ingested, {skipped} unchanged, "
          f"{len(allowlist['rejected'])} excluded by licence")
    print("  " + ", ".join(f"{name} {value}" for name, value in counts.items())
          + f", international names {international}")
    for warning in ingest.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    return 0

