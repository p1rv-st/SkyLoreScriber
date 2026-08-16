"""Tests for the culture catalogue and article assembly.

    python -m unittest discover tests

The licensing tests are the ones that matter most here. `japanese_moon_stations`
carries `![](chart.webp)` inline in all four languages of its prose, and that file is
the single row in `excluded_assets`, admitted on the stated condition that the map is
neither ingested nor served. Returning the article verbatim would breach that, so the
sweep below asserts it over every culture rather than only the known one.
"""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from skylore import cultures, lang

DATABASE = Path(__file__).resolve().parent.parent / "corpus.db"


class StripImages(unittest.TestCase):
    """Unit-level: the rule itself, independent of what the corpus happens to hold."""

    def test_removes_an_excluded_file_even_where_artwork_is_allowed(self):
        text, omitted = cultures.strip_unservable_images(
            "before\n\n![](chart.webp)\n\nafter", images_usable=True,
            excluded={"chart.webp"})
        self.assertNotIn("chart.webp", text)
        self.assertEqual(omitted, [("chart.webp", "excluded_assets")])

    def test_removes_every_image_where_artwork_is_not_allowed(self):
        text, omitted = cultures.strip_unservable_images(
            "![Alt](fine.webp)", images_usable=False, excluded=set())
        self.assertNotIn("fine.webp", text)
        self.assertEqual(omitted, [("fine.webp", "images_usable = 0")])

    def test_keeps_images_a_culture_may_serve(self):
        text, omitted = cultures.strip_unservable_images(
            "![Astronomical building](edificios.webp)", images_usable=True,
            excluded=set())
        self.assertIn("edificios.webp", text)
        self.assertEqual(omitted, [])

    def test_the_two_rules_are_independent(self):
        # They catch the same file today and will not always. A single check would
        # quietly stop covering one case.
        _, by_exclusion = cultures.strip_unservable_images(
            "![](a.webp)", images_usable=True, excluded={"a.webp"})
        _, by_licence = cultures.strip_unservable_images(
            "![](a.webp)", images_usable=False, excluded=set())
        self.assertEqual(by_exclusion[0][1], "excluded_assets")
        self.assertEqual(by_licence[0][1], "images_usable = 0")

    def test_surrounding_prose_survives(self):
        text, _ = cultures.strip_unservable_images(
            "Below is an example.\n\n![](chart.webp)\n\nClose-Up of Yasui's Map",
            images_usable=False, excluded=set())
        self.assertIn("Below is an example.", text)
        self.assertIn("Close-Up of Yasui's Map", text)
        self.assertNotIn("\n\n\n", text)

    def test_text_without_images_is_untouched(self):
        text, omitted = cultures.strip_unservable_images(
            "plain prose", images_usable=False, excluded=set())
        self.assertEqual(text, "plain prose")
        self.assertEqual(omitted, [])


@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class WithCorpus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()


class Licensing(WithCorpus):

    def test_no_article_serves_an_image_it_may_not(self):
        excluded = {(row[0], row[1]) for row in
                    self.db.execute("SELECT culture_id, path FROM excluded_assets")}
        for card in cultures.find_cultures(self.db):
            article = cultures.get_culture_article(
                self.db, card.id, include_boilerplate=True)
            for _alt, path in [m for section in article.sections
                               for m in cultures._IMAGE.findall(section.text)]:
                self.assertTrue(card.images_usable,
                                f"{card.id} serves {path} with images_usable = 0")
                self.assertNotIn((card.id, path), excluded,
                                 f"{card.id} serves excluded asset {path}")

    def test_the_known_excluded_asset_is_gone_and_reported(self):
        article = cultures.get_culture_article(self.db, "japanese_moon_stations")
        self.assertFalse(any("chart.webp" in s.text for s in article.sections))
        self.assertIn(("chart.webp", "excluded_assets"), article.omitted_images)

    def test_it_was_present_in_the_source_to_begin_with(self):
        # If upstream ever drops the reference, the test above starts passing for the
        # wrong reason. This is what keeps it honest.
        present = self.db.execute(
            "SELECT count(*) FROM sections WHERE culture_id = 'japanese_moon_stations'"
            " AND text LIKE '%chart.webp%'").fetchone()[0]
        self.assertGreaterEqual(present, 4, "expected the marker in all four languages")

    def test_cultures_that_may_serve_artwork_still_do(self):
        article = cultures.get_culture_article(self.db, "aztec")
        self.assertTrue(any("![" in s.text for s in article.sections))
        self.assertEqual(article.omitted_images, [])

    def test_attribution_travels_with_every_card_and_article(self):
        for card in cultures.find_cultures(self.db):
            self.assertTrue(card.attribution.strip(), card.id)
            article = cultures.get_culture_article(self.db, card.id)
            self.assertTrue(article.attribution.strip(), card.id)

    def test_licences_are_structured_not_only_prose(self):
        for card in cultures.find_cultures(self.db):
            self.assertTrue(card.text_licenses, card.id)


class Catalogue(WithCorpus):

    def test_returns_every_culture_unranked(self):
        cards = cultures.find_cultures(self.db)
        self.assertEqual(len(cards), 34)
        self.assertEqual([c.id for c in cards], sorted(c.id for c in cards))

    def test_it_fits_in_a_prompt(self):
        # The premise of reading the catalogue instead of searching it.
        cards = cultures.find_cultures(self.db)
        size = sum(len(c.name.value) + len(c.summary.value) + len(c.attribution)
                   for c in cards if c.name and c.summary)
        self.assertLess(size, 40_000, "catalogue has outgrown being read whole")

    def test_every_card_carries_a_name_and_a_summary(self):
        for card in cultures.find_cultures(self.db):
            self.assertIsNotNone(card.name, card.id)
            self.assertIsNotNone(card.summary, card.id)

    def test_name_follows_the_request_and_summary_follows_the_source(self):
        # The two rules applied consistently, which is why they can disagree.
        card = next(c for c in cultures.find_cultures(self.db, locale="ru")
                    if c.id == "anutan")
        self.assertEqual(card.name.lang, "ru")
        self.assertEqual(card.summary.lang, "en")

    def test_region_filter(self):
        asian = cultures.find_cultures(self.db, region="Asia")
        self.assertTrue(asian)
        self.assertTrue(all(c.region == "Asia" for c in asian))

    def test_query_narrows_across_every_language(self):
        # Narrowing with a Russian word must find a culture read in English.
        self.assertTrue(cultures.find_cultures(self.db, query="созвезд"))
        self.assertTrue(cultures.find_cultures(self.db, query="norse"))

    def test_query_matches_the_id_too(self):
        found = cultures.find_cultures(self.db, query="western_rey")
        self.assertEqual([c.id for c in found], ["western_rey"])

    def test_query_that_matches_nothing_returns_nothing(self):
        self.assertEqual(cultures.find_cultures(self.db, query="zzzzz"), [])


class Articles(WithCorpus):

    def test_unknown_culture_is_none_not_an_exception(self):
        self.assertIsNone(cultures.get_culture_article(self.db, "no_such_culture"))

    def test_every_culture_produces_a_non_empty_article(self):
        for card in cultures.find_cultures(self.db):
            article = cultures.get_culture_article(self.db, card.id)
            self.assertTrue(article.sections, card.id)
            self.assertTrue(all(s.text.strip() for s in article.sections), card.id)

    def test_sections_come_back_in_reading_order(self):
        article = cultures.get_culture_article(self.db, "lokono")
        ordinals = [s.ord for s in article.sections]
        self.assertEqual(ordinals, sorted(ordinals))

    def test_boilerplate_is_excluded_by_default(self):
        article = cultures.get_culture_article(self.db, "lokono")
        self.assertFalse({s.kind for s in article.sections} & {"authors", "license"})

        with_boilerplate = cultures.get_culture_article(
            self.db, "lokono", include_boilerplate=True)
        self.assertIn("authors", {s.kind for s in with_boilerplate.sections})

    def test_prose_comes_back_in_the_source_language(self):
        # The decision recorded in PLAN.md §3, visible on every section.
        for locale in ("ru", "es", "zh-Hans"):
            article = cultures.get_culture_article(self.db, "lokono", locale=locale)
            self.assertTrue(all(s.lang == "en" for s in article.sections), locale)

    def test_serving_the_source_means_no_served_section_is_a_fallback(self):
        # 84 rows carry `fallback_from` -- subsections that stayed English inside an
        # otherwise translated article. All of them are translation rows, and serving
        # the source means none is ever returned. So the field is structurally NULL
        # in output today. It is still carried, because that stops being true the
        # moment English stops being total, and a fallback must never be presented
        # as a translation.
        marked = self.db.execute(
            "SELECT count(*) FROM sections WHERE fallback_from IS NOT NULL"
        ).fetchone()[0]
        self.assertGreater(marked, 0, "the ingest-time fallbacks exist")

        served = {(s.lang, s.fallback_from)
                  for card in cultures.find_cultures(self.db)
                  for locale in ("en", "ru", "zh-Hans")
                  for s in cultures.get_culture_article(
                      self.db, card.id, locale=locale).sections}
        self.assertEqual(served, {("en", None)})

    def test_section_selector_takes_a_kind(self):
        article = cultures.get_culture_article(self.db, "lokono", section="introduction")
        self.assertTrue(article.sections)
        self.assertTrue(all(s.kind == "introduction" for s in article.sections))

    def test_section_selector_takes_a_heading_path_prefix(self):
        article = cultures.get_culture_article(
            self.db, "lokono", section="Constellations")
        self.assertTrue(len(article.sections) > 1)
        self.assertTrue(all(s.heading_path.startswith("Constellations")
                            for s in article.sections))

    def test_section_selector_does_not_drag_in_neighbouring_prose(self):
        # "introduction" is a substring of a Description subsection heading in lokono.
        article = cultures.get_culture_article(self.db, "lokono", section="introduction")
        self.assertFalse(any(s.kind == "description" for s in article.sections))

    def test_unknown_section_returns_an_empty_article_not_the_whole_one(self):
        article = cultures.get_culture_article(self.db, "lokono", section="nonesuch")
        self.assertEqual(article.sections, [])
        self.assertTrue(article.attribution.strip(), "attribution still travels")


class Citations(WithCorpus):

    def test_markers_are_resolved_to_their_sources(self):
        article = cultures.get_culture_article(self.db, "anutan")
        cited = [s for s in article.sections if s.references]
        self.assertTrue(cited, "anutan cites references in its prose")
        for section in cited:
            for number, text in section.references.items():
                self.assertIn(f"[#{number}]", section.text)
                self.assertTrue(text.strip())

    def test_only_markers_the_section_actually_uses(self):
        article = cultures.get_culture_article(self.db, "anutan")
        available = self.db.execute(
            "SELECT count(*) FROM section_refs WHERE culture_id = 'anutan' AND lang = 'en'"
        ).fetchone()[0]
        for section in article.sections:
            self.assertLessEqual(len(section.references), available)

    def test_the_references_section_is_not_paired_with_itself(self):
        article = cultures.get_culture_article(self.db, "anutan")
        for section in article.sections:
            if section.kind == "references":
                self.assertEqual(section.references, {})

    def test_no_marker_is_left_unresolved_where_a_source_exists(self):
        from skylore import corpus
        for card in cultures.find_cultures(self.db):
            article = cultures.get_culture_article(self.db, card.id)
            known = {row[0] for row in self.db.execute(
                "SELECT ref_num FROM section_refs WHERE culture_id = ? AND lang = 'en'",
                (card.id,))}
            for section in article.sections:
                if section.kind == "references":
                    continue
                used = corpus.cited_refs(section.text) & known
                self.assertEqual(set(section.references), used,
                                 f"{card.id} {section.heading_path}")


if __name__ == "__main__":
    unittest.main()
