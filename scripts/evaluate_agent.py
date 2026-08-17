"""Judge the agent's prose with an LLM, against the gold set.

    python -m scripts.evaluate_agent --dry-run          # what $0.50 buys, and the ceiling
    python -m scripts.evaluate_agent --budget 0.50      # run it
    python -m scripts.evaluate_agent --ids lang-01 lang-02 --verbose

`scripts.evaluate` scores the *tools*: one tool call per gold question, checking that
the required cultures, stars and constellations came back. That is deterministic, free
and offline -- and it says nothing about the prose the agent writes on top. The prose is
where this project makes most of its promises. Attribution has to be credited in the
answer itself (`skylore.tools` attaches the terms below the agent; only the text can
honour them). A value that fell back to another language has to be *declared* a
fallback, not passed off as a translation. A native name has to arrive as a name, with
its romanisation and meaning, rather than translated away. None of those is a string
comparison, so this module asks a model.

**What the judge is shown, and what it therefore cannot catch.** It gets the question
and the answer, never the tool payloads. That is deliberate: the payloads are what
dominates the cost of a sweep -- 855k characters of corpus JSON across the 68 gold
questions, and one `get_culture_article` answer alone is 76k of it -- so putting them in
front of the judge multiplies the price of every case by the largest term in it. The
judge therefore grades *form* and *coverage against what the gold set says must appear*,
not faithfulness to the sections the tools actually returned. A confident invention in
well-formed prose passes here. Catching that needs the payloads in the judge's context
and costs roughly an order of magnitude more; it is worth doing deliberately, as its own
run, not silently on every sweep.

**Two things are not judged, because `==` decides them.** Whether the model carried the
user's language into its tool calls -- the property `lang-01`/`lang-02` were written for
and the reason `loop._language` states the language rather than injecting it -- and
whether it called a corpus tool at all before answering. Both are read off the
trajectory. Paying a model to answer what a comparison answers is slower, noisier and
dearer, and it would let a judge's mood move a number that has an exact value.

**The bill is bounded before the run, not discovered after it.** `--budget` is not
advice: `UsageLimits(cost_limit=...)` caps each agent run inside the client, the judge is
capped by `max_tokens`, and the number of cases is chosen so that cap times cases stays
under the budget. The worst case is therefore the budget; the actual spend is printed
next to it, because the gap between them is the only honest way to size the next sweep.
A judged number is meaningless without the model that produced it, so the footer names
both models and the date.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic_evals.evaluators import (  # noqa: E402
    EvaluationReason, Evaluator, EvaluatorContext)

from skylore import tools  # noqa: E402
from skylore.agent import checks  # noqa: E402
from skylore.paths import GOLD  # noqa: E402

# What a whole sweep may spend, in dollars. The default is deliberately small: this is a
# number to raise once you have seen what a sweep of ten costs, not a number to discover.
BUDGET = 0.50

# Per agent run, in dollars. Sized from the measured payloads: the dearest gold question
# puts ~25k tokens of article in front of the model, and a run that wants more than this
# is not answering the question -- it is wandering, which is exactly what a ceiling is
# for.
CAP = 0.03

# Per case, for the two judge calls together. The judge sees a question and an answer,
# never a payload -- a couple of thousand tokens in and a verdict out -- but it is a
# larger model than the one under test, at roughly ten times the price per output token,
# so its half of the bill is not a rounding error. It exists so the budget arithmetic
# covers the whole bill rather than the agent's half of it.
JUDGE_ALLOWANCE = 0.02

# Enough for a verdict and two sentences of reason, not enough for an essay -- and not
# so tight that a reasoning model spends the allowance thinking and gets cut off before
# it emits the JSON, which would read as a failed case rather than a truncated judge.
# The effort is low for the same reason it is a judge and not the system under test:
# checking whether a name appears in a paragraph is not a hard problem.
JUDGE_SETTINGS = {"max_tokens": 2000, "openai_reasoning_effort": "low"}

# A judge that is also the model under test grades itself, and models prefer their own
# output. A different family by default; `--judge` overrides.
JUDGE = "openai:gpt-5.2"

PROSE_CONTRACT = """
The output is an answer written for a user whose language is {lang!r}, by a system that
answers only from a corpus of the world's sky cultures. It passes only if all of these
hold:

1. It is written in {lang!r}.
2. It credits the sky cultures whose material it used, naming them in the answer itself.
   An answer that discusses a tradition without ever naming it fails.
3. Native names appear as names -- as they stand, with a romanisation or a meaning
   alongside where one is offered -- rather than being silently replaced by their
   translation.
4. Where it does not have something, it says so plainly instead of hedging or padding.

Judge only these four. Do not reward length, and do not penalise an answer for being
short if it is complete.
""".strip()


# ────────────────────────────────── the environment ──────────────────────────────────

def setting(name: str, fallback: float) -> float:
    """A price ceiling from the environment, or the constant above it.

    Kept lenient on purpose: a typo in `.env` should not stop a sweep that has a
    perfectly good default sitting next to it, and the ceiling actually used is printed
    before anything is spent, so a value that did not take is visible in the first two
    lines of output rather than in the bill.
    """
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return fallback


# ─────────────────────────────────── the cases ───────────────────────────────────

@dataclasses.dataclass
class Question:
    """What the agent is given: the human-shaped question, not the gold set's `probe`.

    The gold set carries both because they measure different things. `probe` is a
    hand-written tool argument and is what `scripts.evaluate` uses; `question` is what a
    person would type, and reaching the right material from it is the agent's job.
    """

    id: str
    text: str
    lang: str
    path: str


# Every requirement handed to the judge is the *name a reader would use*, and never the
# corpus' identifier. The first sweep carried both -- "identifies the star Sirius (HIP
# 32349)" -- with a closing line telling the judge the bracketed part was for reference
# only. It read the parenthesis as part of the requirement anyway and failed answers
# that named Sirius without printing 32349, and failed an answer for saying "Indian
# sky-culture material" rather than "Indian Vedic (indian)". Restating the exemption
# more firmly did not help. A rubric line has to *be* what the answer must contain; an
# id in it is a requirement no matter what the surrounding prose says. Traceability is
# not lost -- the case is named for its gold id, and `data/eval/gold.json` holds the
# expectations in full.

def _culture(connection: sqlite3.Connection, culture_id: str) -> str:
    row = connection.execute(
        "SELECT value FROM culture_names WHERE culture_id = ? AND lang = 'en'",
        (culture_id,)).fetchone()
    return row[0] if row else culture_id


def _star(connection: sqlite3.Connection, hip: int) -> str:
    row = connection.execute(
        "SELECT iau_name, designation FROM stars WHERE hip = ?", (hip,)).fetchone()
    name = (row[0] or row[1]) if row else None
    return name or f"HIP {hip}"


def _constellation(connection: sqlite3.Connection, constellation_id: str) -> str:
    row = connection.execute(
        "SELECT value FROM names WHERE constellation_id = ? AND kind = 'native' "
        "ORDER BY rank LIMIT 1", (constellation_id,)).fetchone()
    return row[0] if row else constellation_id


def rubric(connection: sqlite3.Connection, question: dict) -> str:
    """Turn one gold question's `expect` into something a judge can read.

    The expectations are stored as ids -- `lokono`, `HIP 21421`, `CON tupi 012` -- which
    is right for a database and useless to a judge reading prose: an answer says
    "Aldebaran", not "21421". So every id is resolved to the name a reader would use,
    and the id is kept beside it so a disagreement can be traced back to the row.

    The counting expectations (`min_cultures_naming` and its siblings) survive the trip
    intact, and they are the ones worth having here: "names at least twelve cultures" is
    checkable in prose without any lookup at all.
    """
    expect = question["expect"]
    lines: list[str] = []

    for culture_id in expect.get("cultures", []):
        lines.append(f"names the sky culture {_culture(connection, culture_id)}")
    if expect.get("exact_cultures"):
        lines.append("names those cultures and does not claim any others belong to the "
                     "group the question asks about")
    for hip in expect.get("hips", []):
        lines.append(f"identifies the star {_star(connection, hip)}")
    for constellation_id in expect.get("constellations", []):
        lines.append(f"identifies the constellation "
                     f"{_constellation(connection, constellation_id)}")
    for culture_id, heading in expect.get("sections", []):
        lines.append(f"answers from {_culture(connection, culture_id)}'s article, on the "
                     f"material under {heading!r}")

    for key, what in (("min_cultures_naming", "that name it"),
                      ("min_cultures_drawing", "that draw a figure through it"),
                      ("min_cultures", "")):
        minimum = expect.get(key)
        if minimum is not None:
            lines.append(f"names at least {minimum} distinct sky cultures {what}".strip())

    if expect.get("expect_fallback"):
        lines.append(f"where the corpus has no {question['lang']!r} value and one from "
                     f"another language is shown instead, says so -- a fallback declared "
                     f"as a fallback, never presented as a translation")

    body = "\n".join(f"- {line}" for line in lines)
    return (
        "The output is an answer to the question above, from a corpus of the world's sky "
        "cultures. It passes only if it does all of the following:\n"
        f"{body}\n\n"
        "Judge only this list, and judge it on substance: an answer that names the thing "
        "in its own words satisfies the line, whatever wording it uses. Do not require "
        "any detail the list does not ask for."
    )


def select(gold: dict, budget: float, cap: float, ids: list[str] | None,
           skip: int = 0) -> list[dict]:
    """Which questions the budget pays for.

    Round-robin across the *families* the gold ids name -- `cross-`, `name-`, `lang-`,
    `cat-`, `article-`, `retr-` -- rather than the set's own order, which runs ten star
    questions before the first article. Two reasons for the families over the `path`
    field. A small sweep should touch every tool, and it should touch every *intent*:
    `lang-01` and `name-01` both run `find_constellation`, but only one of them was
    written to ask whether a fallback is declared as a fallback, which is a question
    only this script can answer. Whatever the budget stops at, the sweep spans the set.
    """
    questions = {q["id"]: q for q in gold["questions"]}
    if ids:
        missing = [i for i in ids if i not in questions]
        if missing:
            raise SystemExit(f"no such question: {', '.join(missing)}")
        return [questions[i] for i in ids]

    affordable = int(budget // (cap + JUDGE_ALLOWANCE))
    families: dict[str, list[dict]] = {}
    for question in gold["questions"]:
        families.setdefault(question["id"].rsplit("-", 1)[0], []).append(question)

    ordered: list[dict] = []
    for depth in range(max(len(group) for group in families.values())):
        for group in families.values():
            if depth < len(group):
                ordered.append(group[depth])
    return ordered[skip:skip + affordable]


# ───────────────────────────────────── the run ─────────────────────────────────────

def build_dataset(connection: sqlite3.Connection, questions: list[dict], judge: Any):
    from pydantic_evals import Case, Dataset

    cases = [
        Case(
            name=question["id"],
            inputs=Question(question["id"], question["question"], question["lang"],
                            question["path"]),
            metadata={"why": question["why"]},
            evaluators=(Grounding(rubric=rubric(connection, question), model=judge),),
        )
        for question in questions
    ]
    return Dataset[Question, str, dict](
        name="skylore gold set",
        cases=cases,
        evaluators=[ProseContract(model=judge), LanguageCarried(), UsedTheCorpus()],
    )


def task(model_name: str, provider: str | None, cap: float):
    """The thing under test: one question in, the answer's prose out.

    Only the text is returned. The trajectory and the usage go out of band, as eval
    attributes and metrics, so that the judge's context contains an answer and nothing
    else -- a judge shown the tool calls starts grading the tool calls.
    """
    from pydantic_ai.usage import UsageLimits
    from pydantic_evals import increment_eval_metric, set_eval_attribute

    from skylore.agent import loop

    async def run_agent(question: Question) -> str:
        answer = await loop.ask_async(
            question.text,
            lang=question.lang,
            llm=loop.model(model_name, provider=provider),
            usage_limits=UsageLimits(cost_limit=cap),
        )
        set_eval_attribute("calls", checks.trajectory(answer.calls))
        increment_eval_metric("input_tokens", answer.input_tokens)
        increment_eval_metric("output_tokens", answer.output_tokens)
        if answer.cost is not None:
            increment_eval_metric("agent_usd", round(answer.cost, 6))
        return answer.text

    return run_agent


# ────────────────────────────────── recording a sweep ──────────────────────────────────

# Which evaluator's verdict is whose. The judged two cost money and name their model; the
# trajectory two are exact and free, and are the same checks `skylore.monitor` runs on
# live traffic -- which is the point of writing them to the same table. A pass rate that
# mixes what a model thought with what a comparison proved is not a pass rate.
VERDICTS = {
    "Grounding": ("judge", "grounding"),
    "ProseContract": ("judge", "prose_contract"),
    "LanguageCarried": ("trajectory", "language_carried"),
    "UsedTheCorpus": ("trajectory", "used_the_corpus"),
}


def record(report, *, model_name: str, judge: str, provider: str | None) -> int:
    """Write a finished sweep into the monitoring store. Returns the rows written.

    After the report rather than during it: the sweep's job is to produce a number, and a
    database that is down should cost you the recording, not the sweep you just paid for.
    For the same reason this is a flag and not the default -- `scripts.evaluate_agent`
    has to keep working on a laptop with no Postgres anywhere near it.

    Everything written here comes off the report, so what lands in the database is what
    was printed to the terminal, not a second measurement of the same run.
    """
    from skylore.agent import loop
    from skylore.monitor import store

    store.init()
    written = 0
    for case in report.cases:
        calls = case.attributes.get("calls", [])
        run_id = store.save_run(
            source="eval",
            case_id=case.name,
            question=case.inputs.text,
            lang=case.inputs.lang,
            answer=case.output,
            model=model_name,
            provider=loop.provider_name(provider),
            tools=list(dict.fromkeys(call["tool"] for call in calls)),
            calls=calls,
            input_tokens=int(case.metrics.get("input_tokens", 0)),
            output_tokens=int(case.metrics.get("output_tokens", 0)),
            # Absent when genai-prices did not know the model: unknown, not free.
            cost=case.metrics.get("agent_usd"),
            seconds=case.task_duration,
        )
        for name, assertion in case.assertions.items():
            source, evaluator = VERDICTS.get(name, ("judge", name.lower()))
            store.save_verdict(run_id, source=source, evaluator=evaluator,
                               passed=assertion.value, reason=assertion.reason,
                               model=judge if source == "judge" else None)
        written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    from skylore.agent import loop

    # Before the parser is built, because the file supplies the defaults the flags
    # override -- what a sweep may spend, which model answers, which model judges.
    loop.load_env()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=float, default=setting("SKYLORE_EVAL_BUDGET",
                                                                BUDGET),
                        help=f"dollars the whole sweep may cost; "
                             f"SKYLORE_EVAL_BUDGET, else {BUDGET}")
    parser.add_argument("--cap", type=float, default=setting("SKYLORE_EVAL_CAP", CAP),
                        help=f"dollars one agent run may cost; "
                             f"SKYLORE_EVAL_CAP, else {CAP}")
    parser.add_argument("--ids", nargs="+", help="run these gold questions, ignoring "
                                                 "--budget's arithmetic")
    parser.add_argument("--skip", type=int, default=0,
                        help="skip this many of the rotation, to continue where an "
                             "earlier sweep's budget stopped instead of paying twice")
    parser.add_argument("--model", help=f"the model under test; SKYLORE_MODEL, else "
                                        f"{loop.MODEL}")
    parser.add_argument("--provider", help="overrides SKYLORE_PROVIDER")
    parser.add_argument("--judge", default=os.environ.get("SKYLORE_JUDGE_MODEL", JUDGE),
                        help=f"the judge; SKYLORE_JUDGE_MODEL, else {JUDGE}")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--record", action="store_true",
                        help="write the sweep into the monitoring store, so the judged "
                             "verdicts land in Grafana beside the live traffic; needs "
                             "the `monitor` extra and a Postgres")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the selection, the ceiling and one rubric, and stop")
    parser.add_argument("--verbose", action="store_true",
                        help="print each answer and each judge's reason")
    args = parser.parse_args(argv)

    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    connection = tools.connect()
    questions = select(gold, args.budget, args.cap, args.ids, args.skip)
    ceiling = len(questions) * (args.cap + JUDGE_ALLOWANCE)

    print(f"{len(questions)} of {len(gold['questions'])} questions: "
          f"{', '.join(q['id'] for q in questions)}")
    print(f"ceiling ${ceiling:.2f} = {len(questions)} x (${args.cap:.2f} agent + "
          f"${JUDGE_ALLOWANCE:.2f} judge), budget ${args.budget:.2f}")

    if args.dry_run:
        print(f"\nrubric for {questions[0]['id']} ({questions[0]['question']}):\n")
        print(rubric(connection, questions[0]))
        return 0

    model_name = loop.model_name(args.model)
    judge = metered(args.judge)

    dataset = build_dataset(connection, questions, judge)
    report = dataset.evaluate_sync(
        task(model_name, args.provider, args.cap),
        name=f"skylore agent, judged by {args.judge}",
        max_concurrency=args.concurrency,
    )
    report.print(include_input=False, include_output=args.verbose,
                 include_reasons=args.verbose)

    if args.record:
        written = record(report, model_name=model_name, judge=args.judge,
                         provider=args.provider)
        print(f"\nrecorded {written} runs to the monitoring store "
              f"(source='eval'); see the dashboard at http://localhost:3000")

    agent_cost = sum(case.metrics.get("agent_usd", 0.0) for case in report.cases)
    spent = agent_cost + judge.spent
    print(f"\n{model_name} under test, judged by {args.judge}, "
          f"{dt.date.today().isoformat()}")
    print(f"spent ${spent:.4f} of ${args.budget:.2f} "
          f"(agent ${agent_cost:.4f}, judge ${judge.spent:.4f}); "
          f"${spent / len(questions):.4f} a question")
    remaining = len(gold["questions"]) - len(questions)
    if remaining and spent:
        print(f"the other {remaining} would cost about "
              f"${spent / len(questions) * remaining:.2f} at that rate")
    return 0


# ──────────────────────────── evaluators: the judged half ────────────────────────────

@dataclasses.dataclass
class Grounding(Evaluator[Question, str]):
    """Did the answer contain what the gold set says it must?

    The rubric is per case, built by `rubric()` from that question's `expect`. This is
    the evaluator that makes the gold set worth having twice: `scripts.evaluate` asks
    whether the *tool* returned Aldebaran, and this asks whether the *answer* said so.
    """

    rubric: str
    model: Any = None

    async def evaluate(self, ctx: EvaluatorContext[Question, str]) -> EvaluationReason:
        from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output

        grade = await judge_input_output(
            inputs=ctx.inputs.text, output=ctx.output, rubric=self.rubric,
            model=self.model, model_settings=JUDGE_SETTINGS)
        return EvaluationReason(value=grade.pass_, reason=grade.reason)


@dataclasses.dataclass
class ProseContract(Evaluator[Question, str]):
    """The four promises the answer itself has to keep, whatever the question was.

    Shared across cases and formatted with the user's language, because "answer in the
    user's language" is the one instruction in `loop.INSTRUCTIONS` that the model has to
    honour without any tool helping it.
    """

    model: Any = None

    async def evaluate(self, ctx: EvaluatorContext[Question, str]) -> EvaluationReason:
        from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output

        grade = await judge_input_output(
            inputs=ctx.inputs.text, output=ctx.output,
            rubric=PROSE_CONTRACT.format(lang=ctx.inputs.lang),
            model=self.model, model_settings=JUDGE_SETTINGS)
        return EvaluationReason(value=grade.pass_, reason=grade.reason)


# ───────────────────── evaluators: the half no judge should touch ─────────────────────

# Both of these are `skylore.agent.checks` in an evaluator's clothing. The logic lives
# down there because the monitor runs the same two checks on live traffic, where there is
# no gold set and no judge -- and a check that exists twice stops being one check the
# first time somebody edits a copy.

@dataclasses.dataclass
class LanguageCarried(Evaluator[Question, str]):
    """Every corpus tool call carried the user's language."""

    def evaluate(self, ctx: EvaluatorContext[Question, str]) -> EvaluationReason:
        passed, reason = checks.language_carried(ctx.attributes.get("calls", []),
                                                 ctx.inputs.lang)
        return EvaluationReason(value=passed, reason=reason)


@dataclasses.dataclass
class UsedTheCorpus(Evaluator[Question, str]):
    """The agent asked the corpus before it answered."""

    def evaluate(self, ctx: EvaluatorContext[Question, str]) -> EvaluationReason:
        passed, reason = checks.used_the_corpus(ctx.attributes.get("calls", []))
        return EvaluationReason(value=passed, reason=reason)


# ──────────────────────────────── the judge's own bill ────────────────────────────────

def metered(name: str):
    """The judge model, wrapped so its own spending is counted.

    `judge_input_output` returns a verdict and no usage, so a sweep that only added up
    `Answer.cost` would report the agent's half of the bill as if it were the whole
    thing. Wrapping the model is the one place both judges pass through.
    """
    from pydantic_ai.models import infer_model
    from pydantic_ai.models.wrapper import WrapperModel

    class Metered(WrapperModel):
        spent = 0.0

        async def request(self, messages, model_settings, model_request_parameters):
            response = await super().request(messages, model_settings,
                                             model_request_parameters)
            # `response.cost()` rather than `response.usage.cost`: the field is filled in
            # by the run that owns the response, and the judge's response is handed
            # straight back here before that happens -- which is why a first sweep
            # reported the judge as free. A model the price table does not know raises,
            # and an unpriced judge is not a failed run.
            try:
                type(self).spent += float(response.cost().total_price)
            except LookupError:
                pass
            return response

    return Metered(infer_model(name))


if __name__ == "__main__":
    raise SystemExit(main())
