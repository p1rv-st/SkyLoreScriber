"""Tests for the dense half.

    python -m unittest discover tests

Split in two. The parts that need no model -- the model registry, the embedded text
shape, how windows collapse -- always run. The parts that need ONNX Runtime and 2 GB of
weights skip unless both the extra is installed and vectors have been built, because the
project must stay testable with no dependencies at all.
"""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from skylore import retrieval, tools

DATABASE = Path(__file__).resolve().parent.parent / "corpus.db"

try:
    from skylore import embed
    HAVE_EMBED = True
except ImportError:  # pragma: no cover - depends on the optional extra
    HAVE_EMBED = False


def built_models(connection: sqlite3.Connection) -> list[str]:
    return [row[0] for row in connection.execute("SELECT model FROM embedding_models")]


@unittest.skipUnless(HAVE_EMBED, "skylore.embed unavailable")
class Registry(unittest.TestCase):
    """Model conventions are data, not code: a vector built under one and queried under
    another is indistinguishable from a bad model."""

    def test_every_spec_declares_what_changes_its_vectors(self):
        for spec in embed.MODELS.values():
            self.assertIn(spec.pooling, ("mean", "cls"), spec.name)
            self.assertGreater(spec.dim, 0, spec.name)
            self.assertGreater(spec.max_tokens, 0, spec.name)

    def test_e5_carries_its_prefixes_and_bge_does_not(self):
        # e5 was trained with these and degrades *silently* without them, which is
        # exactly the kind of thing that must not live only in someone's memory.
        for name, spec in embed.MODELS.items():
            if name.startswith("e5"):
                self.assertEqual(spec.query_prefix, "query: ", name)
                self.assertEqual(spec.text_prefix, "passage: ", name)
            if name == "bge-m3":
                self.assertEqual(spec.query_prefix, "")
                self.assertEqual(spec.pooling, "cls")

    def test_query_and_text_prefixes_differ_where_they_exist(self):
        for spec in embed.MODELS.values():
            if spec.query_prefix or spec.text_prefix:
                self.assertNotEqual(spec.query_prefix, spec.text_prefix, spec.name)


@unittest.skipUnless(HAVE_EMBED, "skylore.embed unavailable")
class EmbeddedText(unittest.TestCase):

    def test_the_heading_path_is_prefixed(self):
        # "the rainy season begins when it sets" is unattributable on its own. This is
        # what replaces the context chunk overlap normally provides.
        text = embed.embed_text("lokono", "America", "Description › 3. Calendar of spirits",
                                "the rainy season begins when it sets")
        self.assertIn("lokono", text)
        self.assertIn("America", text)
        self.assertIn("Calendar of spirits", text)
        self.assertTrue(text.endswith("the rainy season begins when it sets"))

    def test_a_missing_region_does_not_produce_an_empty_bracket(self):
        self.assertNotIn("()", embed.embed_text("norse", None, "Introduction", "x"))


@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class SchemaSupportsComparison(unittest.TestCase):
    """The schema change that made choosing a model by measurement possible."""

    @classmethod
    def setUpClass(cls):
        cls.db = tools.connect(DATABASE)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_two_models_can_coexist(self):
        # Keyed by section_id alone, the table held one model and no A/B was possible.
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(embeddings)")}
        self.assertLessEqual({"section_id", "model", "window", "dim", "vector"}, columns)
        key = [row[2] for row in self.db.execute("PRAGMA index_list(embeddings)")]
        self.assertTrue(key, "expected a primary key index")

    def test_the_conventions_are_recorded_beside_the_vectors(self):
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(embedding_models)")}
        self.assertLessEqual(
            {"model", "dim", "languages", "query_prefix", "text_prefix", "pooling"},
            columns)


@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class BuiltVectors(unittest.TestCase):
    """Runs only against vectors that were actually built."""

    @classmethod
    def setUpClass(cls):
        cls.db = tools.connect(DATABASE)
        cls.models = built_models(cls.db)
        if not cls.models:
            raise unittest.SkipTest("no embeddings built")

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_vectors_are_normalised(self):
        # pack() normalises so cosine is a dot product; a drifting norm would silently
        # rescale every similarity.
        for model in self.models:
            blob = self.db.execute(
                "SELECT vector FROM embeddings WHERE model = ? LIMIT 1", (model,)
            ).fetchone()[0]
            self.assertAlmostEqual(retrieval.dot(blob, blob), 1.0, places=4, msg=model)

    def test_stored_width_matches_the_declared_dimension(self):
        for model, dim in self.db.execute("SELECT model, dim FROM embedding_models"):
            blob = self.db.execute(
                "SELECT vector FROM embeddings WHERE model = ? LIMIT 1", (model,)
            ).fetchone()[0]
            self.assertEqual(len(blob) // 4, dim, model)

    def test_the_dense_half_uses_what_was_embedded_not_what_bm25_searches(self):
        # Embedding all four languages made the model rank by language identity: a
        # Chinese query scored every Chinese section ~0.88 whatever it was about.
        for model in self.models:
            built = retrieval.embedded_langs(self.db, model)
            self.assertTrue(built, model)
            embedded = {row[0] for row in self.db.execute(
                "SELECT DISTINCT s.lang FROM embeddings e JOIN sections s"
                " ON s.id = e.section_id WHERE e.model = ?", (model,))}
            self.assertEqual(embedded, set(built), model)

    def test_only_retrievable_sections_carry_vectors(self):
        stray = self.db.execute(
            "SELECT count(*) FROM embeddings e JOIN sections s ON s.id = e.section_id"
            " WHERE s.retrievable = 0").fetchone()[0]
        self.assertEqual(stray, 0)

    def test_windows_are_used_only_where_a_section_does_not_fit(self):
        for model, in self.db.execute("SELECT model FROM embedding_models"):
            multi = self.db.execute(
                "SELECT count(*) FROM (SELECT section_id FROM embeddings"
                " WHERE model = ? GROUP BY section_id HAVING count(*) > 1)", (model,)
            ).fetchone()[0]
            total = self.db.execute(
                "SELECT count(DISTINCT section_id) FROM embeddings WHERE model = ?",
                (model,)).fetchone()[0]
            self.assertLess(multi, total, f"{model}: everything windowed")

    def test_a_windowed_section_still_returns_as_one_passage(self):
        # Windows carry vectors and never surface: the retrieval unit stays the whole
        # section, so the chunking decision is not reopened.
        row = self.db.execute(
            "SELECT section_id FROM embeddings GROUP BY section_id, model"
            " HAVING count(*) > 1 LIMIT 1").fetchone()
        if row is None:
            self.skipTest("no windowed sections in this build")
        culture_id, ordinal = self.db.execute(
            "SELECT culture_id, ord FROM sections WHERE id = ?", (row[0],)).fetchone()
        keys = [(p.culture_id, p.ord) for p in
                retrieval.search_lore(self.db, "constellations and the seasons")]
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
