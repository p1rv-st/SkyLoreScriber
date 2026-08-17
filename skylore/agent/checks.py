"""What a finished run says about itself, without asking a model.

A trajectory answers two questions exactly, and both matter enough to be checked on
every run rather than sampled by a judge:

  * **did the model carry the user's language into its tool calls** -- `loop._language`
    states the language in an instruction instead of injecting it behind the model's
    back, precisely so that this is measurable rather than guaranteed, and the
    `lang-01`/`lang-02` pair in the gold set exists to measure it;
  * **did it ask the corpus at all before answering** -- prose written from the model's
    own memory can be fluent, plausible and entirely outside what this system is allowed
    to say, and a judge shown only the answer cannot tell the difference.

They live here, below both callers, because there are two: `scripts.evaluate_agent`
scores them offline against the gold set, and `skylore.monitor` records them on live
traffic. Two copies of a check are two checks the moment one is edited. Nothing here
imports pydantic-ai or a database -- a trajectory is a list of dictionaries, so the same
function grades a run that just happened and a row read back out of Postgres a week
later.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .. import tools

# `search_web` is deliberately not in `tools.TOOLS`: it is the seventh tool, off unless
# asked for, and the only one that leaves the corpus. Both checks below are about the
# corpus, so membership in this set -- rather than a name spelled out here -- is what
# decides which calls they look at. A tool added to the six is covered automatically.
CORPUS_TOOLS = frozenset(tools.TOOLS)


def trajectory(calls: Iterable[Any]) -> list[dict[str, Any]]:
    """`loop.Call`s as the plain dictionaries everything downstream stores and reads.

    One serialisation, used by the evaluation's attributes and by the monitor's JSONB
    column alike, so a check written against one works on the other. `lang` is lifted out
    of the arguments because it is what `language_carried` asks about; the arguments are
    kept whole beside it because a trajectory is only useful if you can see what was
    actually asked for -- including a malformed call, which `loop.trajectory` records
    rather than drops.
    """
    return [
        {"tool": call.tool,
         "lang": call.arguments.get("lang"),
         "arguments": call.arguments}
        for call in calls
    ]


def language_carried(trace: Iterable[Mapping[str, Any]],
                     lang: str) -> tuple[bool, str]:
    """Every corpus call asked for the user's language. Returns the verdict and why.

    `search_web` takes no `lang` -- it reaches the open web, which has no notion of the
    corpus' locales -- so holding it to this would fail every online run for doing
    exactly what its schema says.
    """
    calls = [call for call in trace if call["tool"] in CORPUS_TOOLS]
    if not calls:
        return False, "no corpus tool was called"

    wrong = [call for call in calls if call.get("lang") != lang]
    if wrong:
        got = ", ".join(sorted({str(call.get("lang")) for call in wrong}))
        return False, (f"{len(wrong)}/{len(calls)} calls passed {got} "
                       f"instead of {lang!r}")
    return True, f"all {len(calls)} calls passed {lang!r}"


def used_the_corpus(trace: Iterable[Mapping[str, Any]]) -> tuple[bool, str]:
    """A corpus tool was called before the answer. Returns the verdict and which tools."""
    used = [call["tool"] for call in trace if call["tool"] in CORPUS_TOOLS]
    return bool(used), ", ".join(dict.fromkeys(used)) or "answered without calling a tool"
