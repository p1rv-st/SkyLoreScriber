"""Web search, off by default, as a seventh tool the agent may not have.

This module exists in tension with the rest of the project, and the tension is the
design. Everything else here answers from a corpus whose provenance is known down to the
licence: 34 cultures, each with an attribution string that travels with every response
because a licence condition a caller has to remember is one that eventually gets
forgotten. The web has none of that. A model that can reach both will, unprompted, blend
them -- and the blended answer is worse than either, because it carries corpus authority
over claims the corpus never made.

Three things follow, and none of them is optional:

**Off unless asked.** `INTERNET_SEARCH` defaults to false and, when false, the tool is
not registered at all rather than registered and refused. A model cannot misuse a tool it
was never shown, and a refusal in a tool result is an invitation to try again differently.

**Results are labelled external in the payload, not only in the prompt.** Every response
carries `external: true` and a `licence` note saying these passages have none of the
corpus's terms. The other tools put provenance in the data for exactly this reason
(`lang.Resolved`, `Passage.matched_lang`); an instruction the model may forget is a
weaker mechanism than a field it has to read past.

**Less reliable, not less useful.** The corpus is 34 cultures out of many, its coverage is
uneven by language (Russian star glosses: 4%), and it can simply be wrong -- a
constellation name mis-transliterated upstream stays mis-transliterated here. So the web
is not restricted to "the corpus returned nothing": filling a gap, adding context around
what the corpus did return, and checking a name that looks wrong are all legitimate. What
is not legitimate is *replacing* the corpus silently. Where the two disagree the answer
has to show both and say which is which, because "the corpus says X, the web says Y" is
information and picking one without saying so destroys it.

No `tavily-python`. The whole client is one POST, and `pydantic-ai` already brings httpx;
declaring httpx and writing the request keeps the returned shape ours, which is the only
way `external: true` gets to be a guarantee rather than a hope.
"""

from __future__ import annotations

import os
from typing import Any

ENDPOINT = "https://api.tavily.com/search"
TIMEOUT = 15.0
MAX_RESULTS = 5

# The corpus half of an answer is licensed; this half is not, and the difference has to
# survive into the payload. Kept here rather than in the tool description because the
# model reads descriptions once and results every time.
LICENCE_NOTE = (
    "External web material, less reliable than the corpus and licensed by nobody here. "
    "Do not credit a sky culture for anything in these results, and do not silently "
    "replace what the corpus returned: where the two disagree, give both and say which "
    "is which. Cite the url."
)

TRUE = frozenset({"1", "true", "yes", "on"})


def enabled(override: bool | None = None) -> bool:
    """Whether the agent gets a web-search tool at all. Default false, deliberately.

    An explicit argument wins so a caller -- the CLI flag, a test -- can turn it on for
    one run without touching the environment.
    """
    if override is not None:
        return override
    return os.environ.get("INTERNET_SEARCH", "").strip().lower() in TRUE


def api_key() -> str:
    """The Tavily key, or a refusal that names what is missing.

    Asked-for-and-broken is a worse outcome than off: if someone sets
    `INTERNET_SEARCH=true` they have said they want the tool, so a missing key is an
    error at startup rather than a tool that silently returns nothing all run.
    """
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "INTERNET_SEARCH is on but TAVILY_API_KEY is not set. Set the key, or "
            "leave INTERNET_SEARCH unset to run on the corpus alone.")
    return key


def _payload(body: dict[str, Any]) -> dict[str, Any]:
    """Tavily's response, reshaped to look like every other tool's output.

    `answer` is deliberately not requested and would be dropped if it arrived: this
    layer's job is to hand the answering model sources, and a second model's synthesis
    inserted between them is one more hop whose provenance nobody can check.
    """
    results = []
    for result in body.get("results", []):
        results.append({
            "title": result.get("title"),
            "url": result.get("url"),
            "text": result.get("content"),
            "score": result.get("score"),
        })
    return {"external": True, "licence": LICENCE_NOTE, "results": results}


async def search(query: str, *, max_results: int = MAX_RESULTS,
                 key: str | None = None, client: Any = None) -> dict[str, Any]:
    """One Tavily search. Errors answer instead of raising, as the other tools do.

    `client` is injectable so the tests exercise the request and the reshaping without a
    network or a key -- there is no other way to test the one thing that can go wrong
    here, which is the shape.
    """
    import httpx

    request = {
        "query": query,
        "max_results": max(1, min(max_results, 10)),
        "search_depth": "basic",
        # No synthesised answer and no raw page dumps: sources, at a readable size.
        "include_answer": False,
        "include_raw_content": False,
    }
    headers = {"Authorization": f"Bearer {key or api_key()}"}

    owned = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT)
    try:
        response = await client.post(ENDPOINT, json=request, headers=headers)
        response.raise_for_status()
        return _payload(response.json())
    except Exception as error:
        # The agent has a corpus to fall back on, so a failed search must not end the
        # run. It has to be visible, though, or the model reports absence of evidence
        # as evidence of absence.
        return {"external": True, "error": f"web search failed: {error}",
                "hint": "answer from the corpus tools, and say the web search failed"}
    finally:
        if owned:
            await client.aclose()


SCHEMA = {
    "name": "search_web",
    "description": (
        "Search the open web, for what the corpus cannot give you or may have got "
        "wrong. Three uses, all legitimate: the corpus covers 34 cultures and returned "
        "nothing on the question; you have a corpus answer and want context around it "
        "(modern astronomy, the sky today, scholarship on the tradition); or a name or "
        "transliteration the corpus returned looks wrong and you want to check it. "
        "Search the corpus first -- these results are less reliable than it, and they "
        "carry no culture's attribution and no licence from this corpus. So: cite the "
        "urls, never credit a sky culture for something that came from here, and where "
        "the web and the corpus disagree, report both and say which said what rather "
        "than choosing one silently."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description":
                      "What to look for, as a full search query."},
            "max_results": {"type": "integer", "description":
                            "Default 5, at most 10."},
        },
        "required": ["query"],
    },
}
