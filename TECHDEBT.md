# Tech debt

Known gaps, recorded when found so they do not get rediscovered as bugs later.
Each entry says what is missing, what breaks because of it, and the cheapest fix.

## Localisation

### 1. Name translation is far patchier than prose translation

Prose is fully translated: every `##` section of all 34 allowed cultures exists in
`ru`, `es` and `zh_CN`. Names are not, and the gap is wide. Measured against the
objects that actually exist in `index.json` (not against the number of catalogue
entries, which is circular):

| what | `en` | `ru` | coverage |
|---|---|---|---|
| constellation glosses | 1529 | 1020 | 66% |
| star glosses | 6093 | 299 | **4%** |

Worst offenders for constellations: `western`, `western_SnT`, `western_hlad` and
`chinese_contemporary` at 0%, `ruelle` at 4%, `western_rey` at 50%. For star names,
`chinese` and `chinese_contemporary` sit at 3% each — and they contribute 6407 of the
6460 star gloss rows, which is why the overall figure is so low.

The IAU-based cultures are the explainable part: Stellarium translates Latin
constellation names centrally rather than per sky culture. The Chinese star names are
simply not in the catalogues.

*Effect:* a Russian answer about Orion or about a named Chinese star returns
translated prose with untranslated names embedded in it. Mixed-language output, worst
on the two cultures a newcomer is likeliest to ask about first.

*Fix:* load the IAU constellation and star name catalogues as a culture-independent
name source, consulted after the per-culture `po` and before falling back to English.
Until then, returning the whole name dictionary and letting the model choose (see
below) is what keeps this from becoming silent breakage.

### 2. No translations at all for international star names

`common_names.tab` sits at the repository root and maps HIP numbers to international
names (`Aldebaran`, `Polaris`, …). There is **no** `po/` directory at the root, so
nothing in this corpus can render those names in another language.

*Effect:* `Альдебаран` is unreachable — neither as a query term nor in output. Users
searching in Russian or Chinese for a star by its common name find nothing, and the
`lookup_star` tool has an English-only entry point.

*Fix:* import the IAU WGSN star name catalogue, which ships native-script and
transliterated forms, and key it by HIP alongside `common_names.tab`. Second-best
option is Wikidata, which has HIP identifiers and multilingual labels.

### 3. Chinese locale variants need an explicit mapping

The corpus ships `zh.po`, `zh_CN.po`, `zh_HK.po`, `zh_TW.po`. A naïve `lang[:2]`
either picks the wrong variant or misses the file entirely.

*Effect:* silently wrong script (Simplified served to a Traditional request) rather
than a visible failure — the worst kind of localisation bug, because it looks fine to
anyone who does not read the language.

*Fix:* a declared variant chain per requested locale, e.g.
`zh-Hant → [zh_TW, zh_HK, zh]` and `zh-Hans → [zh_CN, zh]`, resolved per field and
recorded in `resolved_from_lang`.

## Ingest

### 4. `descritpion` typo in `tukano/index.json`

The per-constellation prose field is spelled `descritpion` in this one culture.
A reader keyed on `description` drops the data with no error.

*Fix:* normalise known field aliases at parse time, and fail loudly on unrecognised
keys in `index.json` so the next upstream typo surfaces instead of vanishing.

### 5. Illustration kinds are not distinguished

`allowlist.json` counts every image under a culture as one number, but there are two
different kinds: constellation artwork under `illustrations/` (has `anchors` in
`index.json`, can be projected onto the sky) and figures referenced inline from the
prose (`lokono_map_1.webp`, `dendera_zodiac.webp`, `Primeros_Memoriales.webp`).
They carry the same licence but belong in different places in the product.

*Also:* `northern_andes` ships `guerrero.webp` at both the culture root and under
`illustrations/`, so the ingest should dedupe by content hash.

### 6. Prose covers more objects than `index.json` does

`lokono` is the clear case: `index.json` has 11 constellations while
`description.md` documents ~33 celestial objects in `#####` sections, including
planets, the Milky Way and constellations whose star patterns are lost. The
structured tables and the prose therefore disagree about what exists.

*Effect:* a purely structured query ("what does this culture know?") understates the
culture, and there is no join between a `#####` prose section and its constellation
row when one exists.

*Fix:* treat the prose sections as their own entity type with an optional link to a
constellation id, rather than assuming every named object has a row in `index.json`.
