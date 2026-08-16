"""Tests for `compare_across_cultures`.

    python -m unittest discover tests

The ranking tests are the ones with teeth. Both obvious orderings are wrong in opposite
directions, and the failure is quiet: a plausible-looking list that buries the best
answers. So the cases that distinguish them are pinned by name, not by position.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from skylore import compare, tools

DATABASE = Path(__file__).resolve().parent.parent / "corpus.db"

ORION = "CON western Ori"
PLEIADES = [17702, 17499, 17573, 17579, 17847, 17851, 18246]


class Scoring(unittest.TestCase):
    """The score alone, with no database."""

    @staticmethod
    def overlap(shared: int, figure: int, target: int) -> compare.Overlap:
        return compare.Overlap(
            constellation_id="x", culture_id="c", names=None,  # type: ignore[arg-type]
            shared_hips=list(range(shared)), figure_size=figure,
            target_size=target, attribution="")

    def test_a_perfect_match_scores_one(self):
        self.assertAlmostEqual(self.overlap(19, 19, 19).score, 1.0)

    def test_a_figure_wholly_inside_the_target_beats_one_merely_crossing_it(self):
        # The measured failure of raw-count ranking: the Egyptian Sah shares 8 stars
        # from a 26-star figure, the Belarusian Throne of Jesus 7 from a 7-star one.
        sah = self.overlap(shared=8, figure=26, target=19)
        throne = self.overlap(shared=7, figure=7, target=19)
        self.assertGreater(sah.shared, throne.shared)
        self.assertGreater(throne.score, sah.score)

    def test_a_near_identical_large_figure_beats_a_tiny_contained_one(self):
        # The opposite failure, from ranking on `of_figure` alone.
        same_orion = self.overlap(shared=17, figure=22, target=19)
        fragment = self.overlap(shared=3, figure=3, target=19)
        self.assertGreater(fragment.of_figure, same_orion.of_figure)
        self.assertGreater(same_orion.score, fragment.score)

    def test_both_fractions_are_reported(self):
        overlap = self.overlap(shared=7, figure=7, target=19)
        self.assertAlmostEqual(overlap.of_figure, 1.0)
        self.assertAlmostEqual(overlap.of_target, 7 / 19)

    def test_no_division_by_zero(self):
        self.assertEqual(self.overlap(0, 0, 0).score, 0.0)


@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class WithCorpus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = tools.connect(DATABASE)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()


class Orion(WithCorpus):

    def setUp(self):
        self.result = compare.compare_across_cultures(
            self.db, constellation_id=ORION, limit=100)

    def test_the_target_is_the_figure_not_the_name(self):
        self.assertEqual(len(self.result.target_hips), 19)

    def test_the_source_figure_is_not_its_own_answer(self):
        self.assertNotIn(ORION, [o.constellation_id for o in self.result.overlaps])

    def test_it_finds_the_reinterpretations_the_corpus_exists_for(self):
        seen = {o.culture_id for o in self.result.overlaps}
        for culture in ("tupi", "navajo", "egyptian", "belarusian", "chinese"):
            self.assertIn(culture, seen)

    def test_results_are_ordered_by_score(self):
        scores = [o.score for o in self.result.overlaps]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def best(self) -> dict[str, tuple[int, compare.Overlap]]:
        """Each culture's best-placed overlap. A culture can draw several figures
        crossing Orion, and keeping the last one seen would compare the wrong pair."""
        found: dict[str, tuple[int, compare.Overlap]] = {}
        for position, overlap in enumerate(self.result.overlaps):
            found.setdefault(overlap.culture_id, (position, overlap))
        return found

    def test_wholly_contained_figures_outrank_the_partial_crossing_one(self):
        # The reordering this ranking exists to produce, on the real rows.
        rank = self.best()
        for contained in ("belarusian", "chinese", "hawaiian_starlines", "korean"):
            self.assertLess(rank[contained][0], rank["egyptian"][0], contained)

    def test_the_partial_one_would_have_won_on_raw_count(self):
        # Otherwise the test above proves nothing about the change.
        rank = self.best()
        self.assertGreater(rank["egyptian"][1].shared, rank["belarusian"][1].shared)
        self.assertLess(rank["egyptian"][1].of_figure, rank["belarusian"][1].of_figure)

    def test_every_overlap_meets_the_threshold(self):
        self.assertTrue(all(o.shared >= compare.MIN_SHARED for o in self.result.overlaps))

    def test_the_threshold_is_doing_work(self):
        loose = compare.compare_across_cultures(
            self.db, constellation_id=ORION, min_shared=1, limit=500)
        self.assertGreater(len(loose.overlaps), len(self.result.overlaps))


class Pleiades(WithCorpus):
    """The bare-HIP entry point: 'who else sees a figure in the Pleiades'."""

    def setUp(self):
        self.result = compare.compare_across_cultures(self.db, hips=PLEIADES, limit=100)

    def test_no_target_constellation_is_required(self):
        self.assertIsNone(self.result.target)
        self.assertEqual(self.result.target_hips, sorted(PLEIADES))

    def test_it_spans_continents(self):
        seen = {o.culture_id for o in self.result.overlaps}
        for culture in ("indian", "boorong", "chinese", "tukano", "inuit", "macedonian"):
            self.assertIn(culture, seen)

    def test_duplicate_hips_do_not_inflate_the_target(self):
        doubled = compare.compare_across_cultures(
            self.db, hips=PLEIADES + PLEIADES, limit=100)
        self.assertEqual(doubled.target_hips, self.result.target_hips)


class Edges(WithCorpus):

    def test_no_stars_gives_an_empty_comparison_not_an_error(self):
        result = compare.compare_across_cultures(self.db, hips=[])
        self.assertEqual(result.target_hips, [])
        self.assertEqual(result.overlaps, [])

    def test_an_unknown_hip_matches_nothing(self):
        self.assertEqual(
            compare.compare_across_cultures(self.db, hips=[999999]).overlaps, [])

    def test_a_constellation_without_a_line_figure(self):
        row = self.db.execute(
            "SELECT id FROM constellations c WHERE NOT EXISTS"
            " (SELECT 1 FROM constellation_lines l WHERE l.constellation_id = c.id)"
            " LIMIT 1").fetchone()
        if row is None:
            self.skipTest("every constellation has lines")
        result = compare.compare_across_cultures(self.db, constellation_id=row[0])
        self.assertEqual(result.target_hips, [])


class Tool(WithCorpus):

    def test_the_tool_is_registered_with_a_schema(self):
        self.assertIn("compare_across_cultures", tools.TOOLS)
        self.assertIn("compare_across_cultures", {s["name"] for s in tools.SCHEMAS})

    def test_it_serialises(self):
        payload = tools.compare_across_cultures(self.db, constellation=ORION, limit=5)
        json.dumps(payload, ensure_ascii=False)
        self.assertEqual(len(payload["seen_elsewhere_as"]), 5)
        self.assertEqual(payload["asked_about"]["star_count"], 19)

    def test_both_shares_reach_the_model(self):
        payload = tools.compare_across_cultures(self.db, constellation=ORION, limit=20)
        for row in payload["seen_elsewhere_as"]:
            self.assertIn("share_of_asked", row)
            self.assertIn("share_of_figure", row)

    def test_the_full_name_dictionary_travels(self):
        # A detail view, per PLAN §3: ~180 strings for a 15-culture comparison, free.
        payload = tools.compare_across_cultures(self.db, constellation=ORION, limit=20)
        row = next(r for r in payload["seen_elsewhere_as"] if r["culture"] == "tupi")
        self.assertTrue(row["names"]["meanings"])

    def test_attribution_travels_for_every_culture_compared(self):
        payload = tools.compare_across_cultures(self.db, constellation=ORION, limit=20)
        mentioned = tools._cultures_mentioned(
            self.db, {k: v for k, v in payload.items() if k != "sources"})
        self.assertTrue(mentioned)
        for culture_id in mentioned:
            self.assertTrue(payload["sources"][culture_id]["attribution"].strip())

    def test_neither_input_answers_rather_than_raising(self):
        result = tools.call(self.db, "compare_across_cultures", {})
        self.assertIn("error", result)
        self.assertIn("hint", result)

    def test_hips_entry_point_through_the_tool(self):
        payload = tools.compare_across_cultures(self.db, hips=PLEIADES, limit=5)
        self.assertNotIn("constellation", payload["asked_about"])
        self.assertTrue(payload["seen_elsewhere_as"])


if __name__ == "__main__":
    unittest.main()
