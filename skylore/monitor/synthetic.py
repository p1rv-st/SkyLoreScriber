"""Plausible traffic with no model behind it.

A dashboard with three rows in it cannot be judged: every panel looks the same whether
its query is right or wrong. Filling the tables with real runs costs money per row and
minutes per hundred, which is a bad way to find out that a `GROUP BY` was wrong.

The questions are the gold set's own -- read from `gold.json` rather than invented --
so the text on the dashboard is the text the system is actually asked, and the tool for
each is the one that question's path implies. The numbers around them are drawn from the
ranges the first real sweeps measured: 2-25k input tokens, 100-900 output, 1.5-9 seconds.
A synthetic row that looked nothing like a real one would make the panels legible and the
axes wrong.

Every row is written with `source='synthetic'`, so nothing here can ever be mistaken for
traffic: `WHERE source = 'chat'` excludes it, and `DELETE FROM runs WHERE source =
'synthetic'` takes it all back out.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

from ..paths import GOLD
from . import store

# The tool a gold question's path implies, so the "which tools get called" panel shows a
# believable spread rather than a uniform one.
TOOL_FOR_PATH = {
    "catalogue": "find_cultures",
    "name": "find_constellation",
    "star": "lookup_star",
    "article": "get_culture_article",
    "retrieval": "search_lore",
}

ANSWER = ("A synthetic answer, recorded so the dashboard has something to draw. "
          "No model was asked and nothing was spent.")


def _questions() -> list[dict]:
    return json.loads(GOLD.read_text(encoding="utf-8"))["questions"]


def one(question: dict, rng: random.Random,
        ts: datetime | None = None) -> int:
    """One synthetic run, with its trajectory verdicts."""
    first = TOOL_FOR_PATH[question["path"]]
    # A second tool on some runs, because real trajectories are not all one call long,
    # and a panel that averages tool count should not be told they are.
    tools = [first] if rng.random() < 0.7 else [first, rng.choice(list(TOOL_FOR_PATH.values()))]

    lang = question["lang"]
    # One run in eight drops the language somewhere, which is roughly what the first
    # sweeps of the real agent did not do -- the panel exists to show the day it starts.
    carried = rng.random() > 0.125
    calls = [{"tool": tool,
              "lang": lang if carried else "en",
              "arguments": {"query": question["probe"], "lang": lang}}
             for tool in tools]

    input_tokens = rng.randint(2_000, 25_000)
    output_tokens = rng.randint(100, 900)

    run_id = store.save_run(
        source="synthetic",
        question=question["question"],
        lang=lang,
        answer=ANSWER,
        model="gpt-5.4-mini",
        provider="openai",
        tools=tools,
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        # The real rates, so the cost panel is in the right units.
        cost=(input_tokens * 0.75 + output_tokens * 4.50) / 1_000_000,
        seconds=rng.uniform(1.5, 9.0),
        ts=ts,
    )

    store.save_verdict(run_id, source="trajectory", evaluator="language_carried",
                       passed=carried, ts=ts,
                       reason=(f"all {len(calls)} calls passed {lang!r}" if carried
                               else f"1/{len(calls)} calls passed 'en' instead of "
                                    f"{lang!r}"))
    store.save_verdict(run_id, source="trajectory", evaluator="used_the_corpus",
                       passed=True, ts=ts, reason=", ".join(dict.fromkeys(tools)))

    # Most people never rate an answer, and the ones who do mostly approve. A generator
    # that thumbed every run would make the panel look solved.
    if rng.random() < 0.35:
        store.save_verdict(run_id, source="user", evaluator="thumbs", ts=ts,
                           score=1 if rng.random() < 0.8 else -1)
    return run_id


def seed(count: int = 200, hours: float = 6.0, seed: int | None = None) -> int:
    """Write `count` synthetic runs spread over the last `hours`. Returns the count.

    Six hours by default because that is Grafana's own default range on the dashboard:
    the point of seeding is to open the dashboard and see it populated, not to open it
    and have to widen the window first.
    """
    rng = random.Random(seed)
    questions = _questions()
    now = datetime.now(timezone.utc)
    for _ in range(count):
        one(rng.choice(questions), rng,
            ts=now - timedelta(seconds=rng.uniform(0, hours * 3600)))
    return count
