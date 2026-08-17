"""Tests for the monitoring layer, run without Postgres, docker or a network.

    python -m unittest discover tests

The whole file skips when `psycopg` is absent, because the project's core installs with
no dependencies at all and the suite has to keep passing in that state.

What is worth asserting here without a server is narrower than it looks, and it is the
half that actually breaks. A wrong column name, a `source` the CHECK constraint rejects,
a value the writer sends that the schema has no place for -- none of these fail at
import, at review, or in any test that mocks the database. They fail on the first insert,
in production, at the moment somebody is watching a dashboard. So the schema is parsed
and the writers are checked against it.

The integration test at the bottom runs against a real Postgres when one answers, and
skips when none does.
"""

from __future__ import annotations

import inspect
import os
import re
import unittest

try:
    import psycopg
except ImportError:  # pragma: no cover - exercised by installing without the extra
    psycopg = None

from skylore import paths

SCHEMA = paths.MONITOR_SCHEMA.read_text(encoding="utf-8")


def columns(table: str) -> set[str]:
    """The column names of one `CREATE TABLE` in `schema.sql`.

    Parsed rather than hard-coded: a list of columns repeated in a test is a list that
    agrees with the schema until the day somebody adds a column to one of them.
    """
    body = re.search(rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\);", SCHEMA,
                     re.S).group(1)
    found = set()
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("--") or line.startswith(")"):
            continue
        name = line.split()[0]
        if name.isidentifier() and name.upper() not in {"CHECK", "PRIMARY", "FOREIGN",
                                                        "UNIQUE", "REFERENCES"}:
            found.add(name)
    return found


def allowed(table: str, column: str) -> set[str]:
    """The values a `CHECK (col IN (...))` permits."""
    body = re.search(rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\);", SCHEMA,
                     re.S).group(1)
    listed = re.search(rf"{column}\s+TEXT NOT NULL CHECK \({column} IN \(([^)]*)\)\)",
                       body).group(1)
    return {value.strip().strip("'") for value in listed.split(",")}


@unittest.skipIf(psycopg is None, "psycopg not installed (uv sync --extra monitor)")
class Schema(unittest.TestCase):
    """The writers and the schema agree, which nothing else checks until an insert."""

    def setUp(self):
        from skylore.monitor import store

        self.store = store

    def test_save_run_writes_columns_that_exist(self):
        known = columns("runs")
        for name in inspect.signature(self.store.save_run).parameters:
            if name == "ts":
                continue  # written as `coalesce(%s, now())`, same column
            with self.subTest(column=name):
                self.assertIn(name, known)

    def test_save_verdict_writes_columns_that_exist(self):
        known = columns("verdicts")
        for name in inspect.signature(self.store.save_verdict).parameters:
            if name in ("run_id", "ts"):
                continue
            with self.subTest(column=name):
                self.assertIn(name, known)

    def test_every_source_the_code_writes_is_permitted(self):
        """`runs.source` and `verdicts.source` are constrained, and the constraint is
        only discovered at insert time."""
        self.assertEqual(allowed("runs", "source"), {"chat", "eval", "synthetic"})
        self.assertEqual(allowed("verdicts", "source"),
                         {"trajectory", "judge", "user"})

    def test_the_recorder_uses_a_permitted_source(self):
        from skylore.monitor import record

        self.assertIn(inspect.signature(record.record).parameters["source"].default,
                      allowed("runs", "source"))

    def test_the_evaluation_records_permitted_sources(self):
        """`scripts.evaluate_agent --record` maps its evaluators onto the same
        vocabulary; a typo there would only surface after a paid sweep."""
        from scripts.evaluate_agent import VERDICTS

        for name, (source, evaluator) in VERDICTS.items():
            with self.subTest(evaluator=name):
                self.assertIn(source, allowed("verdicts", "source"))
                self.assertEqual(evaluator, evaluator.lower())

    def test_the_schema_is_safe_to_apply_twice(self):
        """`store.init()` runs on every app start, so every statement has to tolerate
        already having run."""
        # Comments are stripped before splitting: the prose in this schema contains both
        # semicolons and the word CREATE, and a test that took them for SQL would be
        # asserting things about sentences.
        code = "\n".join(line for line in SCHEMA.splitlines()
                         if not line.strip().startswith("--"))
        statements = [s.strip() for s in code.split(";") if s.strip()]
        self.assertEqual(len(statements), 7)  # two tables, four indexes, one alter
        for statement in statements:
            with self.subTest(statement=statement.splitlines()[0]):
                self.assertIn("IF NOT EXISTS", statement)


@unittest.skipIf(psycopg is None, "psycopg not installed (uv sync --extra monitor)")
class Synthetic(unittest.TestCase):
    """The generator produces rows the schema would accept, without a database."""

    def setUp(self):
        from skylore.monitor import synthetic

        self.synthetic = synthetic
        self.written = []

    def capture(self):
        """Stand in for `store`, keeping what would have been written."""
        import unittest.mock

        def save_run(**kwargs):
            self.written.append(("run", kwargs))
            return len(self.written)

        def save_verdict(run_id, **kwargs):
            self.written.append(("verdict", kwargs))

        return unittest.mock.patch.multiple(self.synthetic.store,
                                            save_run=save_run,
                                            save_verdict=save_verdict)

    def test_rows_are_labelled_synthetic(self):
        """Nothing generated here may ever be mistaken for traffic."""
        import random

        with self.capture():
            self.synthetic.one(self.synthetic._questions()[0], random.Random(0))
        runs = [row for kind, row in self.written if kind == "run"]
        self.assertEqual([run["source"] for run in runs], ["synthetic"])

    def test_every_gold_question_can_be_generated(self):
        """The path-to-tool map has to cover every path the gold set uses."""
        import random

        rng = random.Random(0)
        with self.capture():
            for question in self.synthetic._questions():
                self.synthetic.one(question, rng)
        runs = [row for kind, row in self.written if kind == "run"]
        self.assertEqual(len(runs), len(self.synthetic._questions()))
        for run in runs:
            self.assertTrue(run["tools"])
            self.assertGreater(run["cost"], 0)

    def test_seeding_spreads_the_rows_over_time(self):
        """All in one second makes every time-series panel a single spike."""
        import random

        with self.capture():
            self.synthetic.seed(count=25, hours=6, seed=1)
        stamps = [row["ts"] for kind, row in self.written if kind == "run"]
        self.assertEqual(len(stamps), 25)
        self.assertGreater(max(stamps) - min(stamps),
                           __import__("datetime").timedelta(minutes=30))


@unittest.skipIf(psycopg is None, "psycopg not installed (uv sync --extra monitor)")
class AgainstAPostgres(unittest.TestCase):
    """The round trip, when a server is there to answer. Skipped when none is."""

    @classmethod
    def setUpClass(cls):
        from skylore.monitor import store

        cls.store = store
        try:
            with store.connect():
                pass
        except Exception as error:  # noqa: BLE001 - any failure means "no server"
            raise unittest.SkipTest(f"no Postgres at {os.environ.get('POSTGRES_HOST', 'localhost')}: {error}")
        store.init()

    def test_a_run_and_its_verdicts_come_back(self):
        run_id = self.store.save_run(
            source="synthetic", question="test", lang="en", answer="test",
            model="test-model", provider="openai", tools=["lookup_star"],
            calls=[{"tool": "lookup_star", "lang": "en", "arguments": {}}],
            input_tokens=10, output_tokens=1, cost=None, seconds=0.5)
        self.store.save_verdict(run_id, source="trajectory",
                                evaluator="used_the_corpus", passed=True,
                                reason="lookup_star")
        self.store.save_verdict(run_id, source="user", evaluator="thumbs", score=1)

        row = next(r for r in self.store.recent(50) if r["id"] == run_id)
        self.assertEqual(row["tools"], ["lookup_star"])
        # `cost` is None rather than 0.0: unpriced is not free, and a dashboard that
        # summed it as zero would under-report the bill.
        self.assertIsNone(row["cost"])
        self.assertEqual({v["evaluator"] for v in row["verdicts"]},
                         {"used_the_corpus", "thumbs"})

    def test_an_unknown_source_is_refused_by_the_database(self):
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.store.save_run(
                source="production", question="q", lang="en", answer="a",
                model="m", provider="openai", tools=[], calls=[],
                input_tokens=0, output_tokens=0, cost=None, seconds=0.0)


if __name__ == "__main__":
    unittest.main()
