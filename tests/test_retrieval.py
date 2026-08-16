"""Tests for `search_lore`.

    python -m unittest discover tests

Written against the BM25 half alone, which is what exists until a model is chosen.
Everything here is a property of the plumbing -- deduplication, the per-culture cap,
which language is matched against versus served, and the licensing rules -- rather than
of ranking quality. Ranking is measured by the gold set, not asserted here: a unit test
that pinned today's top-8 would break on every improvement.
"""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from skylore import retrieval, tools

DATABASE = Path(__file__).resolve().parent.parent / "corpus.db"


class Vectors(unittest.TestCase):

    def test_pack_normalises(self):
        packed = retrieval.pack([3.0, 4.0])
        self.assertAlmostEqual(sum(v * v for v in retrieval.unpack(packed)), 1.0, places=5)

    def test_dot_of_a_vector_with_itself_is_one(self):
        packed = retrieval.pack([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(retrieval.dot(packed, packed), 1.0, places=5)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(
            retrieval.dot(retrieval.pack([1.0, 0.0]), retrieval.pack([0.0, 1.0])),
            0.0, places=5)

    def test_a_zero_vector_does_not_divide_by_zero(self):
        self.assertEqual(len(retrieval.unpack(retrieval.pack([0.0, 0.0]))), 2)


class Fusion(unittest.TestCase):

    def test_agreement_beats_a_single_first_place(self):
        # The point of RRF: something both rankers like outranks something only one does.
        fused = dict(retrieval.rrf([(1, 0.0), (2, 0.0)], [(3, 0.0), (2, 0.0)]))
        self.assertGreater(fused[2], fused[1])
        self.assertGreater(fused[2], fused[3])

    def test_one_ranking_is_passed_through_in_order(self):
        fused = retrieval.rrf([(7, 0.0), (8, 0.0), (9, 0.0)])
        self.assertEqual([section for section, _ in fused], [7, 8, 9])

    def test_scores_are_never_compared_across_rankers(self):
        # A BM25 score and a cosine are not comparable numbers. Fusion must depend on
        # position only, so wildly different scales must not change the outcome.
        a = retrieval.rrf([(1, -99.0), (2, -98.0)], [(2, 0.9), (3, 0.1)])
        b = retrieval.rrf([(1, 1.0), (2, 2.0)], [(2, 1000.0), (3, -5.0)])
        self.assertEqual([s for s, _ in a], [s for s, _ in b])

    def test_empty_input(self):
        self.assertEqual(retrieval.rrf(), [])
        self.assertEqual(retrieval.rrf([]), [])


@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class WithCorpus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = tools.connect(DATABASE)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()


class Deduplication(WithCorpus):

    def test_one_section_appears_once_not_once_per_language(self):
        # Each section exists in four languages. Without collapsing before the cut,
        # eight results would be two sections repeated.
        passages = retrieval.search_lore(self.db, "how eclipses of the Sun were explained")
        keys = [(p.culture_id, p.ord) for p in passages]
        self.assertEqual(len(keys), len(set(keys)))

    def test_the_four_language_rows_really_exist(self):
        # The premise of the test above.
        counts = self.db.execute(
            "SELECT count(DISTINCT lang) FROM sections WHERE culture_id = 'mongolian'"
        ).fetchone()[0]
        self.assertEqual(counts, 4)


class Diversity(WithCorpus):

    def test_no_culture_takes_more_than_the_cap(self):
        # lokono is 52k characters over 45 sections against a median culture's 5.7k, so
        # it wins slots by length rather than by relevance. It took 5 of 8 before this.
        for query in ("constellation whose stars are no longer known",
                      "stars and the seasons", "spirits in the sky"):
            passages = retrieval.search_lore(self.db, query)
            counts: dict[str, int] = {}
            for passage in passages:
                counts[passage.culture_id] = counts.get(passage.culture_id, 0) + 1
            self.assertLessEqual(max(counts.values(), default=0), retrieval.PER_CULTURE,
                                 query)

    def test_the_cap_lifts_when_a_culture_is_named(self):
        # "What does this culture say about X" wants everything it has.
        passages = retrieval.search_lore(
            self.db, "constellation whose stars are no longer known", culture="lokono")
        self.assertTrue(all(p.culture_id == "lokono" for p in passages))
        self.assertGreater(len(passages), retrieval.PER_CULTURE)

    def test_capping_actually_changes_this_result(self):
        # Otherwise the cap tests above would pass without the cap existing.
        query = "constellation whose stars are no longer known"
        uncapped = retrieval._cap_per_culture
        try:
            retrieval._cap_per_culture = lambda ranked, cap: ranked
            before = retrieval.search_lore(self.db, query)
        finally:
            retrieval._cap_per_culture = uncapped
        after = retrieval.search_lore(self.db, query)
        self.assertGreater(len({p.culture_id for p in after}),
                           len({p.culture_id for p in before}))


class Languages(WithCorpus):

    def test_a_russian_query_matches_russian_and_is_served_in_english(self):
        # The split built in step 3a, end to end: search_order finds it, prose_order
        # serves it. All four languages are indexed precisely so no query translation
        # is needed.
        passages = retrieval.search_lore(
            self.db, "созвездия и земледельческий календарь", locale="ru")
        self.assertTrue(passages)
        self.assertTrue(all(p.lang == "en" for p in passages))
        self.assertTrue(any(p.matched_lang == "ru" for p in passages),
                        "expected at least one match made against Russian text")

    def test_cross_language_is_reported(self):
        passages = retrieval.search_lore(
            self.db, "созвездия и земледельческий календарь", locale="ru")
        for passage in passages:
            self.assertEqual(passage.cross_language, passage.matched_lang != passage.lang)

    def test_a_spanish_query_finds_spanish_text(self):
        passages = retrieval.search_lore(
            self.db, "constelaciones oscuras formadas por manchas del cielo", locale="es")
        self.assertTrue(passages)
        self.assertTrue(any(p.matched_lang == "es" for p in passages))

    def test_chinese_is_the_known_weak_case(self):
        # unicode61 does not segment CJK, so a run of ideographs is one token and BM25
        # finds nothing. Recorded rather than skipped: when the dense half lands this
        # should start returning results, and that is the signal it is working.
        passages = retrieval.search_lore(
            self.db, "与舞蹈和仪式有关的星座", locale="zh-Hans")
        self.assertEqual(passages, [], "if this now passes, BM25 segments CJK after all")


class Licensing(WithCorpus):

    def test_no_passage_serves_an_image_it_may_not(self):
        excluded = [row[0] for row in self.db.execute("SELECT path FROM excluded_assets")]
        self.assertTrue(excluded)
        for query in ("lunar stations and the seasons", "star chart of the Edo era",
                      "constellations and the agricultural calendar"):
            for passage in retrieval.search_lore(self.db, query):
                for path in excluded:
                    self.assertNotIn(path, passage.text, f"{query} → {passage.culture_id}")

    def test_the_culture_holding_the_excluded_asset_is_reachable_and_clean(self):
        # Reaching it is what makes the test above mean something.
        passages = retrieval.search_lore(self.db, "lunar stations",
                                         culture="japanese_moon_stations")
        self.assertTrue(passages)
        self.assertFalse(any("chart.webp" in p.text for p in passages))

    def test_attribution_travels_through_the_tool(self):
        payload = tools.search_lore(self.db, query="how eclipses were explained")
        mentioned = tools._cultures_mentioned(
            self.db, {k: v for k, v in payload.items() if k != "sources"})
        self.assertTrue(mentioned)
        for culture_id in mentioned:
            self.assertTrue(payload["sources"][culture_id]["attribution"].strip())


class Behaviour(WithCorpus):

    def test_only_retrievable_sections_are_returned(self):
        # Authors and License boilerplate looks alike across every culture and would
        # otherwise match any query about sources.
        for passage in retrieval.search_lore(self.db, "authors and license"):
            self.assertNotIn(passage.kind, ("authors", "license"))

    def test_an_empty_query_returns_nothing(self):
        self.assertEqual(retrieval.search_lore(self.db, "   "), [])

    def test_a_query_of_only_punctuation_returns_nothing(self):
        self.assertEqual(retrieval.search_lore(self.db, "!!! ??? ..."), [])

    def test_fts_operators_are_not_obeyed(self):
        for hostile in ('star OR "', "NEAR(a b)", "star*", "-star", 'a "" b'):
            retrieval.search_lore(self.db, hostile)  # must not raise

    def test_limit_is_respected(self):
        self.assertLessEqual(len(retrieval.search_lore(self.db, "stars", limit=3)), 3)

    def test_citations_are_resolved_when_the_passage_uses_them(self):
        found = False
        for query in ("seafaring and navigation", "the origin of these constellations"):
            for passage in retrieval.search_lore(self.db, query):
                for number in passage.references:
                    self.assertIn(f"[#{number}]", passage.text)
                    found = True
        self.assertTrue(found, "expected at least one cited passage")

    def test_works_without_an_embedder(self):
        # The whole module has to run on BM25 alone, or the baseline is unmeasurable.
        self.assertTrue(retrieval.search_lore(self.db, "eclipses of the Sun",
                                              embedder=None))


if __name__ == "__main__":
    unittest.main()
