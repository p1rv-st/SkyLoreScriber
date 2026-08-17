"""Tests for the judged evaluation, run without a network or an API key.

    python -m unittest discover tests

Everything here is the half of `scripts.evaluate_agent` that costs nothing: the rubric
handed to the judge, the arithmetic that decides how many questions a budget buys, and
the two evaluators that read the trajectory instead of asking a model. The judged half
cannot be tested without spending money, which is precisely why the deterministic half
was kept deterministic.

The rubric tests are not decoration. A first sweep failed answers for naming Sirius
without printing "HIP 32349", because the identifier sat inside the requirement line --
the evaluation was measuring formatting and reporting it as coverage. What a rubric line
says is the whole contract with the judge, so it is asserted like any other output.
"""

from __future__ import annotations

import json
import unittest

try:
    import pydantic_evals  # noqa: F401
except ImportError:  # pragma: no cover - exercised by installing without the extra
    pydantic_evals = None

from skylore import paths, tools

DATABASE = paths.DATABASE


@unittest.skipIf(pydantic_evals is None, "pydantic-evals not installed (uv sync --extra eval)")
@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class Rubrics(unittest.TestCase):
    """What the judge is told to require."""

    @classmethod
    def setUpClass(cls):
        from scripts import evaluate_agent

        cls.evaluate_agent = evaluate_agent
        cls.connection = tools.connect(DATABASE)
        cls.gold = json.loads(paths.GOLD.read_text(encoding="utf-8"))
        cls.questions = {q["id"]: q for q in cls.gold["questions"]}

    @classmethod
    def tearDownClass(cls):
        cls.connection.close()

    def rubric(self, question_id: str) -> str:
        return self.evaluate_agent.rubric(self.connection,
                                          self.questions[question_id])

    def test_expectations_arrive_as_names(self):
        """`cross-04` expects HIP 32349; the judge is asked for Sirius."""
        text = self.rubric("cross-04")
        self.assertIn("Sirius", text)

    def test_no_identifier_reaches_the_judge(self):
        """The regression the first sweep paid to find: an id in a requirement line is a
        requirement, whatever the surrounding prose says it is.

        A star the corpus never named is the one exception -- `HIP 42` is then the only
        handle there is -- which is why this looks for the parenthesised form the rubric
        used to carry beside a name, not for the string itself.
        """
        for question_id, question in self.questions.items():
            with self.subTest(question=question_id):
                text = self.rubric(question_id)
                self.assertNotIn("(HIP", text)
                self.assertNotIn("(CON", text)
                for culture in question["expect"].get("cultures", []):
                    self.assertNotIn(f"({culture})", text)

    def test_every_question_asks_for_something(self):
        """A rubric with no requirements passes anything, which is worse than no case."""
        for question_id in self.questions:
            with self.subTest(question=question_id):
                self.assertIn("- ", self.rubric(question_id))

    def test_a_fallback_question_asks_for_the_declaration(self):
        """`lang-01` exists to check that a fallback is declared rather than passed off
        as a translation, so the rubric has to say so."""
        text = self.rubric("lang-01")
        self.assertIn("fallback", text)


@unittest.skipIf(pydantic_evals is None, "pydantic-evals not installed (uv sync --extra eval)")
class Budget(unittest.TestCase):
    """The arithmetic that decides what a sweep costs, before it costs it."""

    def setUp(self):
        from scripts import evaluate_agent

        self.evaluate_agent = evaluate_agent
        self.gold = json.loads(paths.GOLD.read_text(encoding="utf-8"))

    def select(self, budget, cap=0.03, ids=None, skip=0):
        return self.evaluate_agent.select(self.gold, budget, cap, ids, skip)

    def test_the_budget_bounds_the_selection(self):
        allowance = 0.03 + self.evaluate_agent.JUDGE_ALLOWANCE
        for budget in (0.10, 0.50, 2.00):
            with self.subTest(budget=budget):
                chosen = self.select(budget)
                self.assertLessEqual(len(chosen) * allowance, budget + 1e-9)

    def test_a_small_budget_still_spans_the_set(self):
        """Round-robin over the id families, so nine questions touch all six of them
        rather than nine variations of `lookup_star`."""
        families = {q["id"].rsplit("-", 1)[0] for q in self.select(0.50)}
        self.assertEqual(families,
                         {"cross", "name", "lang", "cat", "article", "retr"})

    def test_skip_continues_where_the_last_sweep_stopped(self):
        first = [q["id"] for q in self.select(0.50)]
        second = [q["id"] for q in self.select(0.50, skip=len(first))]
        self.assertFalse(set(first) & set(second))

    def test_named_questions_win_over_the_arithmetic(self):
        chosen = self.select(0.01, ids=["lang-01", "cross-01"])
        self.assertEqual([q["id"] for q in chosen], ["lang-01", "cross-01"])

    def test_an_unknown_question_is_refused_by_name(self):
        with self.assertRaises(SystemExit) as caught:
            self.select(0.50, ids=["lang-01", "nope-99"])
        self.assertIn("nope-99", str(caught.exception))

    def test_a_ceiling_from_the_environment_is_lenient(self):
        """A typo in `.env` should not stop a sweep that has a good default beside it;
        the ceiling actually used is printed before anything is spent."""
        import os
        import unittest.mock

        with unittest.mock.patch.dict(os.environ, {"SKYLORE_EVAL_BUDGET": "oops"}):
            self.assertEqual(self.evaluate_agent.setting("SKYLORE_EVAL_BUDGET", 0.5), 0.5)
        with unittest.mock.patch.dict(os.environ, {"SKYLORE_EVAL_BUDGET": "0.25"}):
            self.assertEqual(self.evaluate_agent.setting("SKYLORE_EVAL_BUDGET", 0.5),
                             0.25)


@unittest.skipIf(pydantic_evals is None, "pydantic-evals not installed (uv sync --extra eval)")
class Trajectory(unittest.TestCase):
    """The two evaluators that read the tool calls instead of asking a model."""

    def context(self, calls, lang="en"):
        from pydantic_evals.evaluators import EvaluatorContext

        from scripts.evaluate_agent import Question

        return EvaluatorContext(
            name="case", inputs=Question("case", "q", lang, "name"), metadata=None,
            expected_output=None, output="an answer", duration=0.0, _span_tree=None,
            attributes={"calls": calls}, metrics={})

    def test_the_language_has_to_reach_every_call(self):
        from scripts.evaluate_agent import LanguageCarried

        carried = LanguageCarried().evaluate(self.context(
            [{"tool": "lookup_star", "lang": "ru"},
             {"tool": "search_lore", "lang": "ru"}], lang="ru"))
        self.assertTrue(carried.value)

        dropped = LanguageCarried().evaluate(self.context(
            [{"tool": "lookup_star", "lang": "ru"},
             {"tool": "search_lore", "lang": "en"}], lang="ru"))
        self.assertFalse(dropped.value)
        self.assertIn("'ru'", dropped.reason)

    def test_the_web_tool_is_not_held_to_the_corpus_contract(self):
        """`search_web` takes no `lang`; holding it to one would fail every online run
        for doing exactly what its schema says."""
        from scripts.evaluate_agent import LanguageCarried

        result = LanguageCarried().evaluate(self.context(
            [{"tool": "lookup_star", "lang": "en"},
             {"tool": "search_web", "lang": None}]))
        self.assertTrue(result.value)

    def test_an_answer_with_no_tool_call_fails_both(self):
        """Prose written from the model's own memory is the one failure the judge cannot
        see, because the judge is never shown a payload."""
        from scripts.evaluate_agent import LanguageCarried, UsedTheCorpus

        self.assertFalse(UsedTheCorpus().evaluate(self.context([])).value)
        self.assertFalse(LanguageCarried().evaluate(self.context([])).value)

    def test_the_tools_used_are_reported_in_order_without_repeats(self):
        from scripts.evaluate_agent import UsedTheCorpus

        result = UsedTheCorpus().evaluate(self.context(
            [{"tool": "find_cultures", "lang": "en"},
             {"tool": "get_culture_article", "lang": "en"},
             {"tool": "find_cultures", "lang": "en"}]))
        self.assertTrue(result.value)
        self.assertEqual(result.reason, "find_cultures, get_culture_article")


if __name__ == "__main__":
    unittest.main()
