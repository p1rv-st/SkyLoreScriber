"""One finished run, written down and checked.

The join between `skylore.agent` and `skylore.monitor`, and it points one way: this
module imports the agent, the agent knows nothing about this module. A run that is never
recorded is a run that answered anyway.

Two verdicts are written alongside every run, from `agent.checks`. They cost nothing --
no model is asked, the answer is already in hand -- and they are the only quality signal
this system gets on live traffic, where there is no gold set to compare against and no
judge in the request path. `used_the_corpus` in particular is the one check that can
catch an answer composed from the model's own memory, which is the failure this whole
project is built to prevent.
"""

from __future__ import annotations

import time
from typing import Any

from ..agent import checks, loop, web
from . import store


def record(answer: loop.Answer, *, lang: str, seconds: float, model: str,
           source: str = "chat", provider: str | None = None,
           case_id: str | None = None,
           internet: bool = False) -> tuple[int, list[dict[str, Any]]]:
    """Write the run and its trajectory verdicts. Returns the run's id and the verdicts.

    The verdicts come back rather than being merely written, because the caller that
    recorded the run is usually also the one showing it to somebody: the Streamlit app
    prints them under the answer, and reading them back out of Postgres to display what
    it just computed would be a round trip for nothing.
    """
    trace = checks.trajectory(answer.calls)
    run_id = store.save_run(
        source=source,
        question=answer.question,
        lang=lang,
        answer=answer.text,
        model=model,
        provider=provider or loop.provider_name(),
        tools=answer.tools_used,
        calls=trace,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cost=answer.cost,
        seconds=seconds,
        case_id=case_id,
        # Recorded as *offered*, not as used. A run where the web was available and the
        # model stayed in the corpus is the interesting one, and `tools` cannot tell it
        # apart from a run where the web was never on the table.
        internet=internet,
    )

    verdicts = []
    for evaluator, (passed, reason) in (
        ("language_carried", checks.language_carried(trace, lang)),
        ("used_the_corpus", checks.used_the_corpus(trace)),
    ):
        store.save_verdict(run_id, source="trajectory", evaluator=evaluator,
                           passed=passed, reason=reason)
        verdicts.append({"evaluator": evaluator, "passed": passed, "reason": reason})

    return run_id, verdicts


def ask_and_record(question: str, *, lang: str | None = None,
                   model: str | None = None, internet: bool | None = None,
                   images: bool = False,
                   source: str = "chat") -> tuple[loop.Answer, int,
                                                  list[dict[str, Any]], float]:
    """Answer a question and record it, timing the whole run.

    The clock is wall time around the entire agent run -- every model request, every tool
    call, every retry -- because that is the number the person waiting actually
    experiences. `RunUsage` would give a tidier figure that no user has ever felt.
    """
    lang = loop.language(lang)
    name = loop.model_name(model)
    # `None` means "let the environment decide", which is how the rest of the project
    # spells it; resolved here so the recorded flag is what the run actually had rather
    # than what the caller left unsaid.
    online = web.enabled(internet)

    started = time.monotonic()
    answer = loop.ask(question, lang=lang, llm=loop.model(name), internet=online,
                      images=images)
    seconds = time.monotonic() - started

    run_id, verdicts = record(answer, lang=lang, seconds=seconds, model=name,
                              source=source, internet=online)
    return answer, run_id, verdicts, seconds
