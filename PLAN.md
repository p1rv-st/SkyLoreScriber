# Plan

Decisions already taken, and what is left to build. Written to be picked up cold:
each section says what to build, why it is shaped that way, and what is still open.

**Done:** licence scan (`tools/scan_licenses.py` → `allowlist.json`, `licenses.json`),
corpus schema (`skylore/schema.sql`), ingest (`skylore/ingest.py`, `corpus.py`, `po.py`).
`corpus.db` holds 34 cultures, 1529 constellations, 3735 stars, 14731 names,
1304 sections (1020 retrievable), 993 international star names.

**Next:** the query layer (step 3 below). Nothing after it is blocked by anything else.

---

## Build order

| # | step | state | needs |
|---|---|---|---|
| 1 | SQLite schema + structured ingest | done | — |
| 2 | Sections + four materialised languages | done | — |
| 3 | **Query layer: resolver + name lookup** | **next** | nothing |
| 4 | Culture catalogue → `find_cultures`, `get_culture_article` | | 3 |
| 5 | Embeddings + hybrid retrieval → `search_lore` | | 3, 4 |
| 6 | Reranker | only if 5 measures short | 5, gold set |

Steps 3–4 give a working product with no embeddings at all, and answer roughly half
the interesting questions. That is why they come first.

---

## 3. Query layer

### Language resolution

Per **field**, never per document: one missing string must not drop a whole article
to English.

```
resolve_order(L) = variants(L) + [culture.native_lang, "en"]

variants("ru")      = ["ru"]
variants("es")      = ["es"]
variants("zh-Hans") = ["zh_CN", "zh"]
variants("zh-Hant") = ["zh_TW", "zh_HK", "zh"]
```

The variant chain matters because the corpus ships `zh`, `zh_CN`, `zh_HK` and
`zh_TW`, and a naïve `lang[:2]` picks the wrong script silently — the worst kind of
localisation bug, since it looks fine to anyone who cannot read it.

Every resolved value carries the language it actually came from. A row already records
this two ways: `sections.lang` for a whole section, `sections.fallback_from` for a
subsection that stayed English inside an otherwise translated article. Answers must
never present a fallback as a translation.

*Open:* `ruelle` declares `native_lang = "fr"` and `chinese` declares `zh_CN`, but only
`en`/`ru`/`es`/`zh_CN` are ingested, so the `native_lang` step is a no-op for `ruelle`.
Either add `fr` to `TRANSLATION_LANGS` (it is a content source, not a UI language — the
distinction is worth keeping) or drop that step from the chain and say so.

### Names: return the dictionary, mark one preferred

Agreed shape: a detail response carries **every** name variant of the object, and the
model chooses which to use. Two constraints on that:

- The resolver still runs and flags one variant `preferred`, so prose stays internally
  consistent. Without a default the same constellation gets two names in one answer.
- Level of detail decides the volume. Detail views and `compare_across_cultures` get
  the full dictionary (~180 strings for a 15-culture comparison, free). List views get
  `preferred` only — "list the Chinese constellations" is 318 × 12 ≈ 20k tokens
  otherwise.

`native` and `pronounce` must be passed through untranslated and labelled as such, or
the model will "helpfully" render 毕宿 as its gloss and lose the actual name. The
schema enforces this with a CHECK; the tool layer has to preserve it in its output
shape too.

This matters more than it looks: Russian name coverage is 66% for constellation
glosses and **4%** for star glosses (see TECHDEBT.md §1). A resolver that returns one
string would silently pass English off as Russian in almost every star lookup. The
dictionary is what makes that visible.

### Tools: cut by intent, not by mechanism

The tool boundary is the one design decision most likely to be got wrong. Naming tools
`vector_search` / `sql_query` forces the agent to decide "do I need semantic or lexical
retrieval here", which is an information-retrieval question models answer badly.
Naming them by what the user wants works, because picking that is what models are good
at. Mechanism stays inside.

| tool | returns | engine |
|---|---|---|
| `find_cultures(query?, region?, lang)` | the whole catalogue, 34 rows | `culture_summaries`, no search |
| `get_culture_article(id, lang, section?)` | whole article, unchunked | `sections` SELECT |
| `lookup_star(name_or_hip, lang)` | HIP + every culture that names or draws it | `names` + trigram + `star_figures` |
| `find_constellation(name, culture?, lang)` | fuzzy match across all names and languages | trigram FTS |
| `compare_across_cultures(hips[] \| constellation_id, lang)` | the cross-culture join | SQL over `constellation_lines` |
| `search_lore(query, lang, culture?)` | top-8 whole sections | BM25 + dense, RRF |

`lookup_star` must accept every key a user might have: HIP number, international name,
Bayer designation, a native name from any culture, or a translated gloss. That is a
UNION over `stars` and `names` plus trigram fuzzy matching — not one lookup path.

### Two engine facts to honour

- **Trigram FTS matches nothing below three characters.** 237 of 797 native names are
  two-character CJK, so `毕宿` is unreachable via `names_fts`. Queries shorter than
  three characters must fall back to `value LIKE '%q%'` on `names` — a trivial scan at
  14731 rows. Skipping this reads as "search does not work in Chinese".
- **`unicode61` does not segment Chinese or Japanese**, so BM25 over `zh` sections is
  weak and dense retrieval carries that language. Do not expect `sections_fts` to
  perform symmetrically across the four languages.

### Licensing invariants the query layer must enforce

These are not advisory; they are why the licence scan exists.

1. `cultures.attribution` travels with every answer that used that culture.
2. A file in `excluded_assets` is never served. Currently one:
   `japanese_moon_stations/chart.webp`.
3. An illustration is served only where `cultures.images_usable = 1` (15 cultures).
   Artwork licences differ from prose licences — Free Art License is copyleft, so a
   page showing those images must carry LAL terms, not only the prose CC-BY-SA.
4. Everything is non-commercial, so `text_commercial` needs no filtering — but the
   column stays, because that decision could change and `lokono` is CC-BY-NC.

---

## 4. Culture catalogue

`find_cultures` does no searching. With 34 cultures, a catalogue the model reads beats
a similarity search it has to be trusted to get right: 34 rows of
`culture_names` + `culture_summaries` + region + constellation count is ~1500 tokens
whole. Both tables are already populated per language.

`get_culture_article` returns the full article because the corpus is small enough that
it can: ~30k words of prose across all 34 cultures, ~900 words per culture. Handing the
model a whole article removes chunk-boundary loss entirely, and makes the agent pattern
*locate, then read in full* — search as navigation rather than extraction.

---

## 5. Retrieval

Retrieval unit is the **section** (a `###`-level subsection), never smaller. 255
retrievable sections per language, 1020 across the four. Chunking was considered and rejected: at this
corpus size the cost of keeping sections intact is far below the cost of splitting a
markdown table or orphaning a paragraph from its heading.

- **Embed with the heading path prefixed**, e.g.
  `Culture: Lokono (America) — Section: Description › Calendar of spirits`. A section
  saying "the rainy season begins when it sets" is unattributable without it. This
  replaces the context that chunk overlap normally provides.
- **Index all four languages.** BM25 does not work across languages, and materialised
  translations mean a Russian query matches Russian text directly instead of needing
  query translation. This is the payoff of the `po` decision.
- **Multilingual embedder**: `bge-m3` or `multilingual-e5-large`, MPS on this machine.
  Store float32 L2-normalised in `embeddings.vector`; cosine is then a dot product and
  brute force over the ~1020 vectors needs no index and no FAISS.
- **Fuse BM25 and dense with RRF. No reranker at first.** With at most a few hundred
  candidates, return top-8 whole sections (~4k tokens) and let the answering model
  select — at this scale it is a better reranker than a cross-encoder, and cheaper.
  Revisit only against measurements.
- Resolve `[#N]` citations at response time by joining `section_refs` on the markers
  the returned section actually uses. The DB deliberately stores section text verbatim
  rather than substituting inline, so source fidelity is preserved; composing the
  citation is the tool's job. Without this the model receives a bare `[#7]`.

---

## 6. Gold evaluation set

Needed before tuning anything in step 5, or every change is guesswork. ~40 questions,
each recording which culture / section / HIP must appear:

- 1–2 per culture, drawn from that culture's own prose.
- ~10 cross-culture ones ("who else sees a figure in the Pleiades", "which cultures
  name the Hyades") — these test the SQL path, not the vector path.
- A few in each of ru / es / zh, specifically over cultures where name coverage is
  poor, so the fallback surfacing is exercised rather than assumed.
- `lokono` deserves several on its own: 8604 words, the only real ethnographic
  monograph in the corpus, and half of its `## Constellations` section falls back to
  English in translation. If answers there are coherent, the system works.

---

## Open decisions

**Replace `common_names.tab` with the IAU WGSN catalogue.** One change closes three
problems at once: it supplies native-script and multilingual star names (TECHDEBT §1
and §2), and it sidesteps the licence question — `common_names.tab` has no per-culture
licence of its own, so it falls under the repository's AGPL-3.0, which is network
copyleft and would follow the project if it ever became a hosted service. Everything
else we ingest is CC-BY-SA, GPL-2.0 or CC-BY-NC.

**Illustrations in answers.** 330 images across 15 cultures, all permitting
derivatives. Constellation artwork under `illustrations/` carries `anchors` tying it to
HIP positions, so it can be projected onto real sky rather than shown as a flat plate.
Inline figures from the prose (`lokono_map_1.webp`, `dendera_zodiac.webp`) are a
different kind of object and belong in the article. TECHDEBT §5 covers the split.

**Objects that exist only in prose.** `lokono` documents ~33 celestial objects while
its `index.json` has 11 — the rest are planets, the Milky Way, and constellations whose
star patterns are lost. A structured-only answer therefore understates the culture.
These want their own entity type with an optional constellation link, not a pretence
that every named object has a row. TECHDEBT §6.
