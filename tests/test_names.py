"""Tests for name lookup.

    python -m unittest discover tests

These run against the built `corpus.db` rather than a fixture. The behaviours worth
protecting here are properties of the real data -- that two-character CJK names are
reachable, that accented romanisations answer to unaccented queries, that an exact
match outranks 132 substring hits -- and a hand-made fixture would assert them about
data we invented instead of data we ship.
"""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from skylore import lang, names

DATABASE = Path(__file__).resolve().parent.parent / "corpus.db"


@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class NameLookup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()


class ParseHip(unittest.TestCase):

    def test_accepts_the_forms_a_user_would_type(self):
        for text in ("21421", "HIP 21421", "hip21421", "  HIP  21421  "):
            self.assertEqual(names.parse_hip(text), 21421, text)

    def test_rejects_things_that_merely_contain_digits(self):
        for text in ("Wall II", "M45", "", "21421a"):
            self.assertIsNone(names.parse_hip(text), text)


class Keys(NameLookup):
    """`lookup_star` must accept every key someone might hold."""

    def test_hip_number(self):
        for text in ("21421", "HIP 21421", "hip21421"):
            self.assertEqual([s.hip for s in names.lookup_star(self.db, text)], [21421], text)

    def test_unknown_hip_returns_nothing_rather_than_guessing(self):
        self.assertEqual(names.lookup_star(self.db, "HIP 999999"), [])

    def test_international_name(self):
        self.assertEqual(names.lookup_star(self.db, "Aldebaran")[0].hip, 21421)

    def test_bayer_designation_lives_outside_the_names_table(self):
        self.assertEqual(names.lookup_star(self.db, "alf And")[0].hip, 677)

    def test_a_native_name_from_a_culture(self):
        self.assertEqual(names.lookup_star(self.db, "Wara-wara")[0].hip, 21421)

    def test_a_gloss_in_a_non_source_language(self):
        self.assertEqual(
            names.lookup_star(self.db, "Горящий уголь", locale="ru")[0].hip, 21421
        )

    def test_a_bare_number_is_not_searched_as_a_substring(self):
        # "21421" appears inside no name, but the principle matters: an HIP is
        # unambiguous, so it must not compete with text that happens to contain it.
        star = names.lookup_star(self.db, "21421")
        self.assertEqual([s.hip for s in star], [21421])


class TrigramFloor(NameLookup):
    """Queries below three characters form no trigram; `names_fts` returns nothing."""

    def test_two_character_cjk_is_reachable(self):
        # 毕宿 is the case TECHDEBT.md calls out: without the LIKE fallback this
        # returns nothing and search looks broken in Chinese.
        found = names.find_constellation(self.db, "毕宿", locale="zh-Hans")
        self.assertTrue(found)
        self.assertEqual(found[0].id, "CON chinese 001")

    def test_the_index_really_cannot_serve_it(self):
        # If this ever starts returning rows, the fallback is no longer load-bearing
        # and the extra scan can go.
        hits = self.db.execute(
            "SELECT count(*) FROM names_fts WHERE names_fts MATCH '\"毕宿\"'"
        ).fetchone()[0]
        self.assertEqual(hits, 0)

    def test_single_character_names_are_reachable(self):
        self.assertTrue(names.search(self.db, "头", subject="constellation"))

    def test_the_fallback_is_not_a_rare_edge_case(self):
        short = self.db.execute(
            "SELECT count(*) FROM names WHERE length(value) < ?", (names.MIN_TRIGRAM,)
        ).fetchone()[0]
        self.assertGreater(short, 1000)


class Diacritics(NameLookup):
    """`pronounce` rows exist so people can type a romanisation; they are accented."""

    def test_unaccented_query_finds_the_accented_name(self):
        found = names.find_constellation(self.db, "bixiu")
        self.assertEqual(found[0].names.pronounce, "Bìxiù")

    def test_accented_query_still_works(self):
        self.assertEqual(names.find_constellation(self.db, "Bìxiù")[0].id, "CON chinese 001")

    def test_both_spellings_agree(self):
        self.assertEqual(
            [s.hip for s in names.lookup_star(self.db, "carbon encendido", locale="es")],
            [s.hip for s in names.lookup_star(self.db, "Carbón encendido", locale="es")],
        )


class Normalisation(NameLookup):
    """The corpus is inconsistently normalised; a user only ever types one form."""

    def test_the_corpus_really_is_mixed(self):
        # The premise. If upstream ever normalises, the union below stops being
        # load-bearing and can go.
        import unicodedata
        values = [row[0] for row in self.db.execute("SELECT value FROM names")]
        nfd = [v for v in values if v != unicodedata.normalize("NFC", v)]
        nfc = [v for v in values if v != unicodedata.normalize("NFD", v)]
        self.assertTrue(nfd, "expected decomposed rows (Bugis, Mandar)")
        self.assertTrue(nfc, "expected composed rows")

    def test_a_decomposed_name_is_reachable_from_a_composed_query(self):
        # `Bintoѐng Bola Kѐppang` is stored with a combining grave; a keyboard produces
        # the precomposed character. Before the union this matched nothing at all.
        import unicodedata
        stored = self.db.execute(
            "SELECT value FROM names WHERE value LIKE 'Binto%Bola%'").fetchone()[0]
        typed = unicodedata.normalize("NFC", stored)
        self.assertNotEqual(typed, stored, "this test needs the two forms to differ")
        self.assertTrue(names.search(self.db, typed, limit=3))

    def test_a_composed_cyrillic_name_still_matches(self):
        # Regression: normalising the query to NFD alone fixed the Bugis case and broke
        # this one. The tokenizer folds precomposed Latin but not precomposed Cyrillic.
        self.assertTrue(names.search(self.db, "Іллёў воз", limit=3))
        self.assertTrue(names.search(self.db, "Горящий уголь", limit=3))

    def test_both_forms_of_the_same_query_agree(self):
        import unicodedata
        for query in ("Іллёў воз", "Горящий уголь", "Bintoéng Rakkalaé", "Bìxiù"):
            found = {
                tuple(sorted(m.name_id for m in names.search(self.db, form, limit=20)))
                for form in (unicodedata.normalize("NFC", query),
                             unicodedata.normalize("NFD", query))
            }
            self.assertEqual(len(found), 1, f"{query!r} depends on its normal form")

    def test_fold_collapses_the_forms(self):
        import unicodedata
        for text in ("Bìxiù", "Іллёў", "Carbón"):
            self.assertEqual(names.fold(unicodedata.normalize("NFC", text)),
                             names.fold(unicodedata.normalize("NFD", text)))

    def test_exact_matching_survives_normalisation(self):
        # Comparing raw strings would demote a genuine exact match to a substring one
        # whenever the query's form differed from the stored one.
        import unicodedata
        stored = self.db.execute(
            "SELECT value FROM names WHERE value LIKE 'Binto%Bola%'").fetchone()[0]
        matches = names.search(self.db, unicodedata.normalize("NFC", stored), limit=3)
        self.assertEqual(matches[0].tier, "exact")


class Ranking(NameLookup):

    def test_exact_match_outranks_substring(self):
        # "Net" is the exact name of a handful of objects and a substring of ~130.
        matches = names.search(self.db, "Net", subject="constellation", limit=20)
        self.assertEqual(matches[0].tier, "exact")
        tiers = [names.TIERS.index(m.tier) for m in matches]
        self.assertEqual(tiers, sorted(tiers), "tiers must not interleave")

    def test_substring_hits_exist_to_be_outranked(self):
        loose = self.db.execute(
            "SELECT count(*) FROM names_fts WHERE names_fts MATCH '\"net\"'"
        ).fetchone()[0]
        self.assertGreater(loose, 100)

    def test_native_names_sort_with_the_request_not_last(self):
        # native/pronounce carry lang NULL. A `CASE n.lang WHEN ...` would never match
        # NULL and drop them to the bottom; they are the object's own name.
        first = names.search(self.db, "毕宿", limit=5)[0]
        self.assertEqual(first.kind, "native")
        self.assertIsNone(first.lang)


class Filters(NameLookup):

    def test_subject_separates_stars_from_constellations(self):
        self.assertTrue(all(m.hip is not None
                            for m in names.search(self.db, "Net", subject="star")))
        self.assertTrue(all(m.constellation_id is not None
                            for m in names.search(self.db, "Net", subject="constellation")))

    def test_culture_filter(self):
        matches = names.search(self.db, "Net", culture="egyptian")
        self.assertTrue(matches)
        self.assertTrue(all(m.culture_id == "egyptian" for m in matches))

    def test_empty_query_returns_nothing(self):
        self.assertEqual(names.search(self.db, "   "), [])


class QuerySafety(NameLookup):
    """A query is user text, not syntax."""

    def test_fts_operators_are_searched_for_not_obeyed(self):
        for hostile in ('" OR "', "net*", "-net", "NEAR(a b)", 'a "" b'):
            names.search(self.db, hostile)  # must not raise

    def test_like_wildcards_are_escaped(self):
        # Two characters, so this takes the LIKE path where % would otherwise match
        # everything.
        self.assertEqual(names.search(self.db, "%%"), [])


class Dictionaries(NameLookup):
    """What comes back is the object with all its names, not the row that matched."""

    def test_native_and_pronounce_are_kept_out_of_the_glosses(self):
        constellation = names.find_constellation(self.db, "毕宿", locale="zh-Hans")[0]
        self.assertEqual(constellation.names.native, "毕宿")
        self.assertEqual(constellation.names.pronounce, "Bìxiù")
        self.assertNotIn(None, constellation.names.glosses)

    def test_the_whole_dictionary_travels(self):
        star = names.lookup_star(self.db, "Aldebaran")[0]
        bugis = next(n for n in star.names if n.culture_id == "bugis")
        self.assertEqual(set(bugis.glosses), set(lang.LANGUAGES))

    def test_preferred_follows_the_request_not_the_source(self):
        # The round trip this exists to prevent: ask in Chinese, get Chinese back.
        chinese = names.find_constellation(self.db, "毕宿", locale="zh-Hans")[0]
        self.assertEqual(chinese.names.display, "毕宿")
        self.assertEqual(chinese.names.preferred.lang, "zh_CN")

        russian = names.find_constellation(self.db, "Сетка", locale="ru")[0]
        self.assertEqual(russian.names.display, "Сетка")
        self.assertEqual(russian.names.native, "毕宿", "native must survive untranslated")

    def test_display_falls_through_to_native_when_no_gloss_resolves(self):
        star = names.lookup_star(self.db, "Aldebaran")[0]
        boorong = next(n for n in star.names if n.culture_id == "boorong")
        self.assertEqual(boorong.glosses, {})
        self.assertEqual(boorong.display, "Gellarlec")

    def test_a_fallback_is_labelled_rather_than_passed_off_as_a_translation(self):
        # One Russian request over one star exercises both paths: `bugis` has a
        # Russian gloss, `chinese` does not and falls back to English. The point is
        # that the caller can tell which is which -- Russian star-gloss coverage is
        # 4%, so a bare string would be English wearing a Russian label almost
        # everywhere (TECHDEBT.md §1).
        star = names.lookup_star(self.db, "Aldebaran", locale="ru")[0]
        resolved = [n.preferred for n in star.names if n.preferred]

        translated = [r for r in resolved if r.matches_request]
        fell_back = [r for r in resolved if not r.matches_request]
        self.assertTrue(translated, "expected at least one genuine Russian gloss")
        self.assertTrue(fell_back, "expected at least one culture with no Russian gloss")

        self.assertTrue(all(r.lang == "ru" for r in translated))
        self.assertTrue(all(r.is_source for r in fell_back),
                        "a fallback must land on the source language, not a third one")

    def test_international_names_are_grouped_apart_from_cultures(self):
        star = names.lookup_star(self.db, "Aldebaran")[0]
        self.assertIsNone(star.names[0].culture_id)
        self.assertEqual(star.names[0].display, "Aldebaran")


class ConstellationProse(NameLookup):
    """Two upstream sources, and a caller assuming one would lose most of it."""

    def test_index_json_notes_are_returned(self):
        constellation = names.find_constellation(self.db, "Aagjuuk")[0]
        index = [p for p in constellation.prose if p.source == "index"]
        self.assertTrue(index)
        self.assertIn("timekeeping", index[0].text)

    def test_article_subsections_are_returned(self):
        constellation = names.find_constellation(self.db, "Yôkoro wiwa")[0]
        article = [p for p in constellation.prose if p.source == "article"]
        self.assertTrue(article)
        self.assertTrue(article[0].heading_path.startswith("Constellations"))
        self.assertIsNotNone(article[0].section_ord)

    def test_the_two_sources_are_nearly_disjoint(self):
        # 29 constellations have the index note, 23 the article subsection, 1 both.
        # Surfacing only one source would drop roughly half the prose in the corpus.
        with_index = {row[0] for row in self.db.execute(
            "SELECT DISTINCT constellation_id FROM constellation_descriptions")}
        with_article = {row[0] for row in self.db.execute(
            "SELECT DISTINCT constellation_id FROM sections"
            " WHERE constellation_id IS NOT NULL")}
        self.assertTrue(with_index)
        self.assertTrue(with_article)
        self.assertLess(len(with_index & with_article), 5)

    def test_a_constellation_with_both_returns_both(self):
        both = self.db.execute("""
            SELECT constellation_id FROM constellation_descriptions
            INTERSECT
            SELECT constellation_id FROM sections WHERE constellation_id IS NOT NULL
        """).fetchone()
        self.assertIsNotNone(both, "expected at least one constellation with both")
        constellation = names._constellation(self.db, both[0], "en", with_prose=True)
        self.assertEqual({p.source for p in constellation.prose}, {"index", "article"})

    def test_everything_with_prose_in_the_database_is_reachable(self):
        expected = {row[0] for row in self.db.execute("""
            SELECT constellation_id FROM constellation_descriptions
            UNION
            SELECT constellation_id FROM sections WHERE constellation_id IS NOT NULL
        """)}
        reachable = {
            cid for (cid,) in self.db.execute("SELECT id FROM constellations")
            if names._constellation(self.db, cid, "en", with_prose=True).prose
        }
        self.assertEqual(reachable, expected)

    def test_prose_resolves_to_the_source_language_unlike_the_names(self):
        constellation = names.find_constellation(self.db, "Aagjuuk", locale="ru")[0]
        self.assertTrue(all(p.lang == "en" for p in constellation.prose))

    def test_a_list_view_does_not_carry_prose(self):
        # Aldebaran is drawn by 27 cultures; prose for each would swamp the answer.
        star = names.lookup_star(self.db, "Aldebaran")[0]
        self.assertGreater(len(star.figures), 20)
        self.assertTrue(all(figure.prose == [] for figure in star.figures))

    def test_no_prose_passage_serves_an_image(self):
        # A regression guard, not proof of the mechanism: no linked passage carries an
        # image today, so this passes trivially. The rule itself is exercised in
        # test_cultures.StripImages; the reason it is wired in here anyway is that
        # assuming "this text has no markup" is what let chart.webp through once.
        served = [p for (cid,) in self.db.execute("SELECT id FROM constellations")
                  for p in names._constellation(self.db, cid, "en", with_prose=True).prose]
        self.assertTrue(served)
        for passage in served:
            self.assertNotIn("![", passage.text)


class Licensing(NameLookup):

    def test_attribution_travels_with_every_constellation(self):
        for query in ("毕宿", "Orion", "Gellarlec"):
            for constellation in names.find_constellation(self.db, query):
                self.assertTrue(constellation.attribution.strip(), constellation.id)

    def test_attribution_travels_through_a_star_lookup_too(self):
        star = names.lookup_star(self.db, "Aldebaran")[0]
        self.assertTrue(star.figures)
        for figure in star.figures:
            self.assertTrue(figure.attribution.strip(), figure.id)


class CrossCulture(NameLookup):

    def test_a_star_reports_the_cultures_that_draw_through_it(self):
        star = names.lookup_star(self.db, "HIP 21421")[0]
        cultures = {figure.culture_id for figure in star.figures}
        self.assertGreater(len(cultures), 5, "Aldebaran is drawn by many cultures")

    def test_naming_and_drawing_are_different_relations(self):
        # A culture can draw a figure through a star without naming the star, and
        # name a star it draws no figure through. Both must be reported.
        star = names.lookup_star(self.db, "HIP 21421")[0]
        named = {n.culture_id for n in star.names if n.culture_id}
        drawn = {figure.culture_id for figure in star.figures}
        self.assertTrue(named)
        self.assertTrue(drawn - named, "expected a culture that draws but does not name")


if __name__ == "__main__":
    unittest.main()
