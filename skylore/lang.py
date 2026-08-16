"""Which stored language a request maps onto.

Two separate questions live here, and conflating them into one "current language"
is the mistake the module exists to prevent:

  * **Search** -- which languages to match a query against. Wide. A Russian query
    should hit the Russian text directly, which is the entire reason four languages
    are materialised, but it must still reach a Bayer designation or an HIP number
    that only ever appears in the English rows.

  * **Prose** -- which language to hand an article or section back in. Narrow, and
    it is the *source* text. The answering model translates and synthesises for the
    user itself, so serving it a corpus translation only adds a second lossy hop:
    upstream already rendered English into Russian, and the model would then render
    that Russian again. English is the source in the literal sense --
    `description.md` is written in it and every `po` catalogue is keyed on English
    msgids (see `po.py`), so every other language here is downstream of it.

  * **Names** -- the opposite order, and the reason `prose_order` and `name_order`
    are two functions rather than one flag. A name is not prose to be rendered; it
    is a term the corpus already holds in the user's language, and re-deriving it
    from English is both wasteful and wrong. Asked in Chinese about 毕宿, an answer
    must say 毕宿 -- not translate the query to "Net", search, and translate "Net"
    back, which is a different word by the time it returns. So the stored `zh_CN`
    gloss outranks the English one.

    The `native` row is not part of this ordering. It is always returned alongside,
    untranslated, and becomes the preferred name only when no gloss resolves at all
    -- a Chinese speaker asking about a Norse figure wants the Chinese gloss with the
    Norse name beside it, not the Norse name as the answer.

Measured against the built corpus, English is total: all 326 sections, all 1529
constellation glosses, and zero translated glosses lacking an English counterpart.
The output chain therefore almost never leaves its first step. It is still written
as a chain, because upstream adds cultures and a future missing English gloss must
degrade to another language rather than to nothing.

Native names never enter any of this. `names.kind IN ('native', 'pronounce')` is
stored with `lang IS NULL` -- 毕宿 is not English awaiting translation, it is the
name -- and the schema's CHECK enforces it. Callers pass those through untouched.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

# English is the source language; the rest are materialised from the po catalogues.
SOURCE_LANG = "en"
TRANSLATION_LANGS = ("ru", "es", "zh_CN")
LANGUAGES = (SOURCE_LANG, *TRANSLATION_LANGS)

# ─────────────────────────────── locale variants ───────────────────────────────

# Upstream ships zh, zh_CN, zh_HK and zh_TW. `zh` is Simplified in practice, so
# truncating a tag to its bare language -- the obvious `lang[:2]` -- answers a
# Traditional request in Simplified. That failure is invisible to anyone who cannot
# read the script, which is why the chains below are declared rather than derived.
# TECHDEBT.md §3.
_ZH_CHAINS: dict[str | None, tuple[str, ...]] = {
    "Hans": ("zh_CN", "zh"),
    "Hant": ("zh_TW", "zh_HK", "zh"),
    None:   ("zh", "zh_CN", "zh_TW", "zh_HK"),
}
_ZH_BY_REGION = {"CN": "Hans", "SG": "Hans", "MY": "Hans",
                 "TW": "Hant", "HK": "Hant", "MO": "Hant"}
_ZH_REGIONAL = {"zh_CN", "zh_TW", "zh_HK"}  # region forms upstream actually ships

# language[-script][-region], separated by either "-" (BCP 47) or "_" (gettext).
_TAG = re.compile(
    r"^([A-Za-z]{2,3})(?:[-_]([A-Za-z]{4}))?(?:[-_]([A-Za-z]{2}|[0-9]{3}))?(?:[-_].*)?$"
)


def _parse(locale: str) -> tuple[str, str | None, str | None]:
    """Split a locale tag into (language, script, region), normalised to gettext case.

    Anything unparseable is treated as a bare language rather than raising: a locale
    is user input, and an odd tag should narrow the search, not fail the request.
    """
    match = _TAG.match(locale.strip())
    if match is None:
        return locale.strip().lower(), None, None
    language, script, region = match.groups()
    return (
        language.lower(),
        script.title() if script else None,
        region.upper() if region else None,
    )


def _dedupe(tags: tuple[str, ...]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for tag in tags:
        seen.setdefault(tag, None)
    return tuple(seen)


def variants(locale: str) -> tuple[str, ...]:
    """Corpus language tags a request could legitimately be served from, best first.

    Ordering is by specificity, and it never widens across a script boundary:

        variants("ru")      == ("ru",)
        variants("pt-BR")   == ("pt_BR", "pt")
        variants("zh-Hans") == ("zh_CN", "zh")
        variants("zh-Hant") == ("zh_TW", "zh_HK", "zh")

    The result is unfiltered -- it says what would be acceptable, not what exists.
    the callers below intersect it with what the database actually holds.
    """
    language, script, region = _parse(locale)

    if language == "zh":
        if script is None and region is not None:
            script = _ZH_BY_REGION.get(region)
        chain = _ZH_CHAINS.get(script, _ZH_CHAINS[None])
        exact = f"zh_{region}"
        return _dedupe((exact, *chain) if exact in _ZH_REGIONAL else chain)

    if script is not None:
        # Deliberately no bare-language step: dropping a script subtag is the
        # zh-Hant bug in its general form (sr-Latn served as Cyrillic). A language
        # whose scripts we can order safely gets a declared chain above instead.
        return _dedupe((f"{language}_{script}_{region}", f"{language}_{script}")
                       if region else (f"{language}_{script}",))

    return _dedupe((f"{language}_{region}", language) if region else (language,))


# ──────────────────────────────── resolution ────────────────────────────────

def search_order(locale: str, available: tuple[str, ...] = LANGUAGES) -> tuple[str, ...]:
    """Languages to match a query against, most promising first.

    Every available language, not only the requested one. The request leads because
    matching Russian against Russian needs no query translation -- the payoff of
    materialising the po catalogues -- but a query is often a proper noun, an HIP
    number or a Bayer designation that appears in exactly one language's rows, so
    restricting the search to the requested language loses those outright.

    Callers use the order for ranking and de-duplication, not for filtering.

    Search crosses a script boundary where the two output orders must not, and the
    asymmetry is deliberate. Serving Simplified text to a Traditional request is a
    silent defect; *matching* a Traditional query against Simplified rows only costs
    some precision, and refusing it drops Chinese out of the search altogether,
    since zh_CN is the only Chinese this database holds.
    """
    preferred = [lang for lang in variants(locale) if lang in available]
    base = _parse(locale)[0]
    related = [lang for lang in available
               if lang not in preferred and _parse(lang)[0] == base]
    rest = [lang for lang in available if lang not in preferred and lang not in related]
    return tuple(preferred + related + rest)


def prose_order(locale: str, available: tuple[str, ...] = LANGUAGES) -> tuple[str, ...]:
    """Languages to return an article or section in, best first: source before
    translation.

    The requested language still ranks above the remaining ones, so that a section
    with no English text degrades toward whoever asked rather than arbitrarily.

    Not for names -- see `name_order`.
    """
    order = [SOURCE_LANG] if SOURCE_LANG in available else []
    order += [lang for lang in variants(locale) if lang in available and lang not in order]
    order += [lang for lang in available if lang not in order]
    return tuple(order)


def name_order(locale: str, available: tuple[str, ...] = LANGUAGES) -> tuple[str, ...]:
    """Languages to mark one name variant `preferred` in, best first: request before
    source.

    Inverted against `prose_order` on purpose. Prose gets translated downstream
    whatever we send, so the source loses least; a name does not survive that trip.
    "毕宿" glossed to "Net" and rendered back arrives as some other Chinese word,
    and the corpus already holds the right one -- for all 318 Chinese
    constellations, as both a `zh_CN` gloss and an untranslated `native` row.

    This only chooses the *preferred* variant. Detail responses carry the whole name
    dictionary regardless (PLAN.md §3), because Russian gloss coverage is 4% on stars
    and one string cannot both be complete and be short.
    """
    order = [lang for lang in variants(locale) if lang in available]
    if SOURCE_LANG in available and SOURCE_LANG not in order:
        order.append(SOURCE_LANG)
    return tuple(order + [lang for lang in available if lang not in order])


@dataclass(frozen=True)
class Resolved:
    """A value together with the language it actually came from.

    The provenance is the point. A resolver that returned a bare string would pass
    English off as Russian in almost every star lookup -- Russian gloss coverage is
    4% there (TECHDEBT.md §1) -- and the caller could not tell. Answers must never
    present a fallback as a translation, so every resolved value carries its origin.
    """
    value: str
    lang: str
    requested: str

    @property
    def is_source(self) -> bool:
        """True when this is the corpus's own text rather than a translation of it."""
        return self.lang == SOURCE_LANG

    @property
    def matches_request(self) -> bool:
        """True when the value is in a language the request would have accepted."""
        return self.lang in variants(self.requested)


def pick(by_lang: dict[str, str], order: tuple[str, ...], requested: str) -> Resolved | None:
    """First non-empty value in `order`, tagged with where it came from.

    Resolution is per field, never per document: one missing string must not drop a
    whole article to another language, so callers resolve each field through this
    rather than choosing a language once and reading everything in it.
    """
    for lang in order:
        value = by_lang.get(lang)
        if value:
            return Resolved(value=value, lang=lang, requested=requested)
    return None


def available_langs(connection: sqlite3.Connection) -> tuple[str, ...]:
    """Languages this database was actually built with, source language first.

    Read rather than assumed: `LANGUAGES` describes what the current ingest writes,
    and a database built by an older one would otherwise resolve to rows that are
    not there.
    """
    rows = connection.execute("SELECT DISTINCT lang FROM sections").fetchall()
    langs = {row[0] for row in rows}
    return tuple(
        [SOURCE_LANG] * (SOURCE_LANG in langs) + sorted(langs - {SOURCE_LANG})
    )
