-- SkyLoreScriber corpus schema.
--
-- Design notes that are not obvious from the DDL:
--
--  * Only cultures on the allowlist are ingested. `skipped_cultures` records the rest
--    so an absence is always explained rather than looking like a parsing failure.
--
--  * A name has three roles, and they behave differently under translation.
--    `native` is what the culture itself calls the object (毕宿, Aña, Aagjuuk) and never
--    varies by interface language. `pronounce` is its romanisation, likewise fixed.
--    Only `gloss` -- the meaning, rendered in some language -- is translatable. The
--    CHECK constraint enforces this: a gloss must name its language, the other two
--    must not. Translating a native name would destroy data, so it is prevented here
--    rather than left to callers to remember.
--
--  * Nothing is pre-resolved. Every language version that exists in the source is
--    stored; choosing which to show when one is missing is the query layer's job, so
--    the fallback chain stays testable code instead of baked-in rows.
--
--  * `constellation_lines.hip` is the join that makes cross-culture questions possible
--    ("who else sees a figure in the Pleiades"), hence the index on it.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ─────────────────────────── cultures and provenance ───────────────────────────

CREATE TABLE cultures (
    id                  TEXT PRIMARY KEY,
    region              TEXT,
    classification      TEXT,             -- JSON array, e.g. ["incomplete"]
    native_lang         TEXT,             -- locale of the native script, NULL if none
    highlight           TEXT,             -- constellation the source suggests showing first
    constellation_count INTEGER NOT NULL,

    -- Licensing, copied from allowlist.json so the database answers licence questions
    -- on its own. Prose and artwork are licensed separately in most cultures.
    text_licenses       TEXT    NOT NULL, -- JSON array
    text_commercial     INTEGER NOT NULL,
    text_share_alike    INTEGER NOT NULL,
    image_licenses      TEXT    NOT NULL, -- JSON array, empty when there is no artwork
    images_usable       INTEGER NOT NULL,

    attribution         TEXT    NOT NULL, -- Authors section verbatim; every answer carries it
    source_sha256       TEXT    NOT NULL  -- of description.md + index.json, for incremental re-ingest
);

CREATE TABLE culture_names (
    culture_id TEXT NOT NULL REFERENCES cultures(id) ON DELETE CASCADE,
    lang       TEXT NOT NULL,
    value      TEXT NOT NULL,
    PRIMARY KEY (culture_id, lang)
);

-- One or two sentences per culture per language, taken from the Introduction section.
-- `find_cultures` returns this table wholesale: at 34 cultures a catalogue the model
-- reads beats a similarity search it has to be trusted to get right.
CREATE TABLE culture_summaries (
    culture_id TEXT NOT NULL REFERENCES cultures(id) ON DELETE CASCADE,
    lang       TEXT NOT NULL,
    summary    TEXT NOT NULL,
    PRIMARY KEY (culture_id, lang)
);

CREATE TABLE skipped_cultures (
    id      TEXT PRIMARY KEY,
    reasons TEXT NOT NULL                 -- JSON array, from allowlist.json
);

-- Files a manual licence review carved out. Serving one of these is a licence breach,
-- so the constraint travels with the data.
CREATE TABLE excluded_assets (
    culture_id TEXT NOT NULL,
    path       TEXT NOT NULL,
    reason     TEXT NOT NULL,
    PRIMARY KEY (culture_id, path)
);

-- ──────────────────────────────── sky objects ────────────────────────────────

-- Every HIP the corpus mentions, whether or not it has an international name, so
-- that constellation lines and cultural star names always have something to point at.
CREATE TABLE stars (
    hip         INTEGER PRIMARY KEY,
    iau_name    TEXT,                     -- primary international name, English
    designation TEXT,                     -- Bayer/Flamsteed, e.g. "alf And"
    named_by    TEXT                      -- provenance of iau_name
);

CREATE TABLE constellations (
    id         TEXT PRIMARY KEY,          -- "CON norse 001"
    culture_id TEXT    NOT NULL REFERENCES cultures(id) ON DELETE CASCADE,
    ord        INTEGER NOT NULL,          -- order within the culture's index.json
    iau        TEXT,                      -- IAU code where the culture maps onto one
    image_path TEXT,
    image_w    INTEGER,
    image_h    INTEGER
);
CREATE INDEX ix_constellations_culture ON constellations(culture_id);

-- Short prose attached to a constellation. Two upstream sources land here: the
-- `description` field of index.json, and `##### <native name>` subsections of
-- description.md. Both are translated through the same po entries.
CREATE TABLE constellation_descriptions (
    constellation_id TEXT NOT NULL REFERENCES constellations(id) ON DELETE CASCADE,
    lang             TEXT NOT NULL,
    text             TEXT NOT NULL,
    PRIMARY KEY (constellation_id, lang)
);

-- A figure is a set of polylines, not a set of stars: `segment` distinguishes the
-- strokes and `seq` orders the vertices within one, so the shape survives the round trip.
CREATE TABLE constellation_lines (
    constellation_id TEXT    NOT NULL REFERENCES constellations(id) ON DELETE CASCADE,
    segment          INTEGER NOT NULL,
    seq              INTEGER NOT NULL,
    hip              INTEGER NOT NULL REFERENCES stars(hip),
    -- Line weight ("thin"/"bold") where the source specifies one, on seq = 0 only:
    -- it is a property of the whole segment. Used by western and western_SnT.
    style            TEXT,
    PRIMARY KEY (constellation_id, segment, seq)
);
CREATE INDEX ix_lines_hip ON constellation_lines(hip);

CREATE TABLE image_anchors (
    constellation_id TEXT    NOT NULL REFERENCES constellations(id) ON DELETE CASCADE,
    idx              INTEGER NOT NULL,
    x                INTEGER NOT NULL,
    y                INTEGER NOT NULL,
    hip              INTEGER NOT NULL REFERENCES stars(hip),
    PRIMARY KEY (constellation_id, idx)
);

-- ─────────────────────────────────── names ───────────────────────────────────

CREATE TABLE names (
    id               INTEGER PRIMARY KEY,
    constellation_id TEXT REFERENCES constellations(id) ON DELETE CASCADE,
    hip              INTEGER REFERENCES stars(hip),
    culture_id       TEXT REFERENCES cultures(id) ON DELETE CASCADE,  -- NULL = culture-independent
    kind             TEXT    NOT NULL,
    lang             TEXT,
    value            TEXT    NOT NULL,
    rank             INTEGER NOT NULL DEFAULT 0,  -- source order; 0 is the primary name

    CHECK (kind IN ('native', 'pronounce', 'gloss')),
    -- Exactly one subject.
    CHECK ((constellation_id IS NULL) <> (hip IS NULL)),
    -- Only a gloss carries a language; see the header note.
    CHECK ((kind = 'gloss') = (lang IS NOT NULL))
);
CREATE INDEX ix_names_constellation ON names(constellation_id);
CREATE INDEX ix_names_hip           ON names(hip);
CREATE INDEX ix_names_culture       ON names(culture_id);
CREATE INDEX ix_names_value         ON names(value COLLATE NOCASE);

-- Trigram rather than word tokens: names are short, often in scripts a word
-- tokeniser mishandles (毕宿 does not tokenise under unicode61 at all), and callers
-- need substring and near-miss matching more than they need relevance ranking.
--
-- Callers must handle one limit: a query shorter than three characters forms no
-- trigram and matches nothing here. That is not a rare edge case -- 237 of the
-- corpus's native names are two-character CJK -- so a search of fewer than three
-- characters has to fall back to `value LIKE '%q%'` against `names`, which is a
-- trivial scan at this table's size.
CREATE VIRTUAL TABLE names_fts USING fts5(
    value,
    content = 'names',
    content_rowid = 'id',
    tokenize = 'trigram'
);

-- ────────────────────────────────── prose ──────────────────────────────────

-- The retrieval unit: one subsection of one culture's article in one language.
-- Deliberately not chunked further -- the whole corpus is ~40k tokens, so the cost
-- of keeping sections intact is far lower than the cost of losing context at a
-- chunk boundary.
CREATE TABLE sections (
    id               INTEGER PRIMARY KEY,
    culture_id       TEXT    NOT NULL REFERENCES cultures(id) ON DELETE CASCADE,
    lang             TEXT    NOT NULL,
    ord              INTEGER NOT NULL,   -- reading order within the article
    kind             TEXT    NOT NULL,   -- introduction | description | extras | constellations | references | other
    level            INTEGER NOT NULL,   -- markdown heading depth, 2..6
    heading          TEXT,               -- own heading in `lang`; NULL for a section's lead text
    heading_path     TEXT    NOT NULL,   -- "Description › Calendar of spirits"; prefixed onto the
                                         -- embedded text so a retrieved section is self-describing
    text             TEXT    NOT NULL,
    constellation_id TEXT REFERENCES constellations(id) ON DELETE SET NULL,
    -- Set when this row fell back to another language because `lang` was missing
    -- upstream. NULL means the text really is in `lang`. Answers must not present a
    -- fallback as a translation, so the provenance is per row, not per article.
    fallback_from    TEXT,
    retrievable      INTEGER NOT NULL DEFAULT 1,  -- 0 for Authors/License boilerplate

    UNIQUE (culture_id, lang, ord)
);
CREATE INDEX ix_sections_lookup ON sections(culture_id, lang, kind);

-- Resolved [#N] footnotes. Without these a retrieved section hands the model a bare
-- "[#7]" and the citation is lost.
CREATE TABLE section_refs (
    culture_id TEXT    NOT NULL REFERENCES cultures(id) ON DELETE CASCADE,
    lang       TEXT    NOT NULL,
    ref_num    INTEGER NOT NULL,
    text       TEXT    NOT NULL,
    PRIMARY KEY (culture_id, lang, ref_num)
);

-- unicode61 does not segment Chinese or Japanese, so BM25 over zh sections is weak
-- and dense retrieval carries those languages. Recorded in TECHDEBT.md.
CREATE VIRTUAL TABLE sections_fts USING fts5(
    heading_path,
    text,
    content = 'sections',
    content_rowid = 'id',
    tokenize = "unicode61 remove_diacritics 2"
);

CREATE TABLE embeddings (
    section_id INTEGER PRIMARY KEY REFERENCES sections(id) ON DELETE CASCADE,
    model      TEXT    NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB    NOT NULL           -- float32, L2-normalised: cosine is a dot product
);

-- ────────────────────────────────── views ──────────────────────────────────

-- Which cultures draw a figure through a given star. The cross-culture question,
-- pre-joined.
CREATE VIEW star_figures AS
SELECT DISTINCT l.hip, c.culture_id, c.id AS constellation_id
FROM constellation_lines l
JOIN constellations c ON c.id = l.constellation_id;
