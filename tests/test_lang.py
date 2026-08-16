"""Tests for the language resolver.

    python -m unittest discover tests

Weighted toward the failures that are invisible rather than loud: serving the wrong
Chinese script, and presenting a fallback as a translation. Both look correct to
anyone who does not read the language, so they have to be caught here or not at all.
"""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from skylore import lang

DATABASE = Path(__file__).resolve().parent.parent / "corpus.db"


class Variants(unittest.TestCase):

    def test_plain_language_is_itself(self):
        self.assertEqual(lang.variants("ru"), ("ru",))
        self.assertEqual(lang.variants("es"), ("es",))

    def test_region_narrows_before_widening(self):
        self.assertEqual(lang.variants("pt-BR"), ("pt_BR", "pt"))
        self.assertEqual(lang.variants("ru_RU"), ("ru_RU", "ru"))

    def test_both_separators_accepted(self):
        self.assertEqual(lang.variants("zh-Hans"), lang.variants("zh_Hans"))

    def test_case_is_normalised_to_gettext_form(self):
        self.assertEqual(lang.variants("ZH-HANT"), lang.variants("zh-Hant"))
        self.assertEqual(lang.variants("pt-br"), ("pt_BR", "pt"))

    def test_simplified_and_traditional_do_not_cross(self):
        # The TECHDEBT.md §3 bug: `lang[:2]` maps both onto `zh`, which upstream
        # ships as Simplified, so a Traditional request is silently answered in the
        # wrong script.
        hans = lang.variants("zh-Hans")
        hant = lang.variants("zh-Hant")
        self.assertEqual(hans[0], "zh_CN")
        self.assertEqual(hant[0], "zh_TW")
        self.assertNotIn("zh_CN", hant)
        self.assertNotIn("zh_TW", hans)
        self.assertNotIn("zh_HK", hans)

    def test_script_inferred_from_region(self):
        self.assertNotIn("zh_CN", lang.variants("zh-TW"))
        self.assertNotIn("zh_CN", lang.variants("zh-HK"))
        self.assertEqual(lang.variants("zh-SG")[0], "zh_CN")

    def test_exact_regional_variant_leads(self):
        self.assertEqual(lang.variants("zh-HK")[0], "zh_HK")
        self.assertEqual(lang.variants("zh-Hant-HK")[0], "zh_HK")
        # …but the rest of the Traditional chain still follows it.
        self.assertEqual(lang.variants("zh-HK"), ("zh_HK", "zh_TW", "zh"))

    def test_bare_chinese_prefers_the_exact_catalogue(self):
        self.assertEqual(lang.variants("zh")[0], "zh")

    def test_script_is_never_truncated_away(self):
        # Widening sr-Latn to sr would serve Cyrillic to a Latin request -- the
        # zh-Hant bug in its general form.
        self.assertNotIn("sr", lang.variants("sr-Latn"))
        self.assertEqual(lang.variants("sr-Latn-RS"), ("sr_Latn_RS", "sr_Latn"))

    def test_unparseable_tag_degrades_instead_of_raising(self):
        self.assertEqual(lang.variants("not a locale"), ("not a locale",))

    def test_no_duplicates(self):
        for tag in ("zh", "zh-CN", "zh-Hans-CN", "zh-TW", "ru", "pt-BR"):
            chain = lang.variants(tag)
            self.assertEqual(len(chain), len(set(chain)), tag)


class SearchOrder(unittest.TestCase):

    def test_requested_language_leads(self):
        self.assertEqual(lang.search_order("ru")[0], "ru")

    def test_search_crosses_a_script_boundary_that_output_will_not(self):
        # zh_CN is the only Chinese stored, so a Traditional query has no exact
        # variant to match. Ranking it below English would leave Chinese search
        # answered by languages that share none of its characters.
        self.assertEqual(lang.search_order("zh-Hant")[0], "zh_CN")
        self.assertNotEqual(lang.prose_order("zh-Hant")[0], "zh_CN")
        self.assertNotEqual(lang.name_order("zh-Hant")[0], "zh_CN")

    def test_covers_every_available_language(self):
        # A query is often a proper noun or a Bayer designation living in exactly one
        # language's rows, so search must never be filtered down to the request.
        for tag in ("ru", "en", "zh-Hant", "de"):
            self.assertEqual(set(lang.search_order(tag)), set(lang.LANGUAGES), tag)

    def test_unstored_language_still_searches_everything(self):
        self.assertEqual(set(lang.search_order("de")), set(lang.LANGUAGES))


class ProseOrder(unittest.TestCase):

    def test_source_leads_regardless_of_request(self):
        # The decision recorded in PLAN.md §3: the answering model translates and
        # synthesises itself, so a corpus translation on the way in is a second
        # lossy hop over text that is already downstream of the English.
        for tag in ("ru", "es", "zh-Hans", "de"):
            self.assertEqual(lang.prose_order(tag)[0], "en", tag)

    def test_request_outranks_the_remaining_languages(self):
        order = lang.prose_order("ru")
        self.assertLess(order.index("ru"), order.index("es"))

    def test_degrades_toward_the_request_when_source_is_absent(self):
        order = lang.prose_order("ru", available=("ru", "es", "zh_CN"))
        self.assertEqual(order[0], "ru")

    def test_covers_every_available_language(self):
        self.assertEqual(set(lang.prose_order("ru")), set(lang.LANGUAGES))


class NameOrder(unittest.TestCase):

    def test_request_leads_and_inverts_prose(self):
        # A name asked for in Chinese must come back in Chinese. Glossing it to
        # English and rendering that back returns a different word.
        self.assertEqual(lang.name_order("zh-Hans")[0], "zh_CN")
        self.assertEqual(lang.prose_order("zh-Hans")[0], "en")

    def test_source_is_the_fallback_not_a_tail_language(self):
        # With no Chinese name, English is next -- not Russian or Spanish, which
        # are themselves translations of the English and no closer to the request.
        self.assertEqual(lang.name_order("zh-Hant")[:2], ("en", "ru"))
        self.assertEqual(lang.name_order("de")[0], "en")

    def test_covers_every_available_language(self):
        self.assertEqual(set(lang.name_order("ru")), set(lang.LANGUAGES))


class Pick(unittest.TestCase):

    def test_returns_the_first_available(self):
        resolved = lang.pick({"en": "Orion", "ru": "Орион"}, lang.prose_order("ru"), "ru")
        self.assertEqual(resolved.value, "Orion")
        self.assertEqual(resolved.lang, "en")

    def test_reports_that_a_fallback_is_not_a_translation(self):
        resolved = lang.pick({"en": "Orion"}, lang.prose_order("ru"), "ru")
        self.assertTrue(resolved.is_source)
        self.assertFalse(resolved.matches_request)

    def test_reports_a_genuine_match(self):
        resolved = lang.pick({"ru": "Орион"}, ("ru", "en"), "ru")
        self.assertTrue(resolved.matches_request)
        self.assertFalse(resolved.is_source)

    def test_matches_request_understands_variant_chains(self):
        resolved = lang.pick({"zh_CN": "猎户"}, ("zh_CN",), "zh-Hans")
        self.assertTrue(resolved.matches_request)
        wrong_script = lang.pick({"zh_CN": "猎户"}, ("zh_CN",), "zh-Hant")
        self.assertFalse(wrong_script.matches_request)

    def test_empty_strings_are_skipped_not_returned(self):
        # An empty msgstr is how gettext spells "untranslated"; treating it as a hit
        # would return a blank field and call it a translation.
        resolved = lang.pick({"en": "", "ru": "Орион"}, lang.prose_order("ru"), "ru")
        self.assertEqual(resolved.lang, "ru")

    def test_nothing_available_is_none_not_an_exception(self):
        self.assertIsNone(lang.pick({}, lang.prose_order("ru"), "ru"))


@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class AgainstTheCorpus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)

    @classmethod
    def tearDownClass(cls):
        cls.connection.close()

    def test_available_langs_matches_what_was_ingested(self):
        # Order past the source language is alphabetical rather than LANGUAGES order:
        # what matters is the set, and a deterministic tail that does not depend on
        # the ingest constant the database may predate.
        self.assertEqual(set(lang.available_langs(self.connection)), set(lang.LANGUAGES))

    def test_source_language_leads_what_the_database_reports(self):
        self.assertEqual(lang.available_langs(self.connection)[0], "en")

    def test_a_chinese_name_is_served_from_chinese_rows(self):
        # The round trip this module exists to prevent: query in Chinese, gloss to
        # English, search, translate back. Every Chinese constellation holds its own
        # name, so resolution lands on it directly and never leaves the language.
        available = lang.available_langs(self.connection)
        by_lang = dict(self.connection.execute(
            "SELECT n.lang, n.value FROM names n JOIN constellations c"
            " ON c.id = n.constellation_id"
            " WHERE c.culture_id = 'chinese' AND c.ord = 0 AND n.kind = 'gloss'"
        ).fetchall())

        resolved = lang.pick(by_lang, lang.name_order("zh-Hans", available), "zh-Hans")
        self.assertEqual(resolved.lang, "zh_CN")
        self.assertEqual(resolved.value, "毕宿")
        self.assertTrue(resolved.matches_request)
        # …while the article about it still comes back in the source language.
        self.assertEqual(lang.prose_order("zh-Hans", available)[0], "en")

    def test_every_chinese_constellation_has_a_chinese_name(self):
        missing = self.connection.execute("""
            SELECT count(*) FROM constellations c
             WHERE c.culture_id = 'chinese'
               AND NOT EXISTS (SELECT 1 FROM names n
                                WHERE n.constellation_id = c.id
                                  AND ((n.kind = 'gloss' AND n.lang = 'zh_CN')
                                       OR n.kind = 'native'))
        """).fetchone()[0]
        self.assertEqual(missing, 0)

    def test_english_is_total_over_glosses(self):
        # The premise the output chain rests on: every translated gloss has an
        # English counterpart, so resolution never leaves its first step in practice.
        # If upstream ever breaks this, the chain is what saves the field -- but the
        # docstring claiming it is total should stop claiming that.
        orphans = self.connection.execute("""
            SELECT count(*) FROM (
                SELECT constellation_id, hip, culture_id FROM names
                 WHERE kind = 'gloss' AND lang <> 'en'
                EXCEPT
                SELECT constellation_id, hip, culture_id FROM names
                 WHERE kind = 'gloss' AND lang = 'en')
        """).fetchone()[0]
        self.assertEqual(orphans, 0)

    def test_native_names_carry_no_language_to_resolve(self):
        stray = self.connection.execute(
            "SELECT count(*) FROM names WHERE kind IN ('native', 'pronounce') "
            "AND lang IS NOT NULL"
        ).fetchone()[0]
        self.assertEqual(stray, 0)

    def test_every_stored_language_is_reachable_from_a_plausible_request(self):
        requests = {"en": "en", "ru": "ru", "es": "es-419", "zh_CN": "zh-Hans"}
        for stored, request in requests.items():
            self.assertIn(stored, lang.variants(request), f"{request} -> {stored}")


if __name__ == "__main__":
    unittest.main()
