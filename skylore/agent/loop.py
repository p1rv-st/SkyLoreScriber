"""The RAG loop: a PydanticAI agent over the six tools, and two optional ones.

This is the first module in the project that cannot run offline, which is why
`pydantic-ai-slim` lives in the `agent` extra. Everything below it still installs with
no dependencies at all.

The six are always registered. `search_web` is added when the deployment allows it and
`show_constellation_image` when the caller has a screen to draw on; both are left out
entirely rather than registered and refused, and each brings its own paragraph of
instructions only when its tool is present.

Three decisions are worth reading before changing anything here.

**The model sees `tools.SCHEMAS`, not docstrings written twice.** Those descriptions are
the most carefully argued text in the project -- they are what makes a tool cut by
intent rather than by mechanism (PLAN.md §3) -- so they are rendered into the wrapper
docstrings rather than paraphrased beside them. `tests/test_agent.py` sweeps the
signatures against the schemas, because the parameter names come from the signature
while their descriptions come from the schema, and a drift between the two loses a
description silently.

**The wrappers are thin on purpose.** `lang` defaults to `"en"` here exactly as it does
in `skylore.tools`, and the agent is *told* the user's language in an instruction rather
than having it injected behind its back. That keeps one contract instead of two, and it
makes "did the model carry the language through" a measurable property of the run --
which is what the `lang-01`/`lang-02` pair in the gold set was built to check. If
measurement says the model drops it, the fix belongs here and should be recorded.

**Licensing survives because it was never the agent's job.** `_with_sources` attaches
attribution inside `skylore.tools`, below this layer, so a tool wrapper cannot forget
it. What this layer adds is the instruction to *use* it: terms that reach the model and
die there satisfy nothing.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sqlite3
from typing import Any

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.usage import UsageLimits

from .. import paths, tools
from ..query import retrieval
from . import web

MODEL = "gpt-5.4-mini"
LANG = "en"

# Pictures one answer may display. Two, because a picture beside a paragraph illustrates
# it and six bury it -- and because the schema's polite version of this rule was measured
# and found to hold only until somebody asks for a gallery.
IMAGE_LIMIT = 2

# Chat Completions rather than the Responses API: OpenRouter speaks the former, and
# nothing here needs the latter. One transport for both providers means the answer to
# "does this work on OpenRouter" is not a separate code path that nobody exercises.
PROVIDERS = ("openai", "openrouter")


# ──────────────────────────────── what the deployment sets ────────────────────────────
#
# Three things belong to whoever runs this rather than to the code: which model answers,
# which language it answers in, and -- with `SKYLORE_PROVIDER`, which was here first --
# where the requests go. Each is resolved the same way, and in the same order: an
# explicit argument wins, then the environment, then the constant above. Resolved on
# call rather than read into a constant at import, because `.env` is loaded by the entry
# point and a constant would be fixed before that ever happened.

def load_env(path=None) -> None:
    """Put `.env` into the environment, without overwriting anything already set.

    Called by the entry points -- the CLI and the evaluation scripts -- and by nothing
    below them. Importing a library should not read files off disk or decide which model
    a caller meant; a program that embeds `skylore.agent` sets its own environment and
    this stays out of its way. The caller's own exports win over the file for the same
    reason.
    """
    path = path or paths.ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def model_name(name: str | None = None) -> str:
    """Which model answers. `SKYLORE_MODEL` overrides the default, `--model` overrides
    both -- a number from an evaluation means nothing without the model that produced
    it, so the model is a setting and not a constant to edit."""
    return name or os.environ.get("SKYLORE_MODEL") or MODEL


def language(lang: str | None = None) -> str:
    """The language the answer and the names come back in. `SKYLORE_LANG` sets the
    deployment's default; `--lang` sets it per question.

    It is a default and not a policy: `lang` travels into the run as an instruction the
    model has to carry into its own tool calls, which is what `lang-01`/`lang-02` and
    `LanguageCarried` in the evaluation exist to check.
    """
    return lang or os.environ.get("SKYLORE_LANG") or LANG


# ──────────────────────────────────── the loop ────────────────────────────────────

@dataclasses.dataclass
class Deps:
    """What the tools need and the model must not choose: a connection, a language, and
    whether the dense half of `search_lore` is available.

    `lang` is not passed to the tools from here. It is rendered into an instruction, so
    the model carries it through its own tool calls and the run records whether it did.
    """

    connection: sqlite3.Connection
    lang: str = "en"
    embedder: retrieval.Embedder | None = None

    # How many pictures have actually been displayed in this run. Mutable, and the only
    # mutable thing here, because the limit below is per answer and there is nowhere else
    # that knows what "this answer" means.
    images_shown: int = 0


INSTRUCTIONS = """
You answer questions about the sky cultures of the world -- 34 traditions, their
constellations, their star names and the stories attached to them -- using only the
tools provided. The corpus is the authority; you are not. If the tools do not have
something, say so plainly rather than filling the gap from general knowledge, and say
which cultures you did check.

Choosing a tool:
- The question names a star, however the user has it (a name, a HIP number, a Bayer
  designation, a meaning): lookup_star.
- It names a constellation or a figure: find_constellation.
- It asks who else sees something in the same stars, or how other traditions read a
  pattern: compare_across_cultures. This is the question the corpus exists to answer.
- It is about one culture generally: find_cultures to locate it, then
  get_culture_article to read it. Articles come back whole -- read one properly rather
  than sampling several.
- It names no star, constellation or culture at all ("how were eclipses explained"):
  search_lore.

Reading what comes back:
- Attribution is not optional. Every response carries a `sources` block with the terms
  each culture's material is under. Credit each culture you actually drew on, in the
  answer itself.
- Prose is returned in English because English is the language it was written in.
  Translate it yourself into the user's language. Do not ask a tool for a translated
  article -- there is not one.
- `native` and `pronounce` are the name itself, not English awaiting translation. Give
  the native name as it stands, with its romanisation and meaning alongside.
- `is_fallback: true` means the corpus had no value in the language asked for and this
  one came from another. Say so; do not present it as a translation. Russian star-gloss
  coverage is 4%, so this will be common.
- An `omitted` block means material was removed under its licence. Do not mention those
  figures, describe them, or ask for them again.
- Naming and drawing are different relations. A culture can draw a figure through a star
  it never names, and name a star it draws no figure through. Do not merge the two.

Answer in the user's language, in prose. Be concrete: name the culture, and name what
it calls the thing.
""".strip()

# Added only when the tool is registered. An instruction describing a tool the model
# cannot see is worse than no instruction: it invites the model to explain what it would
# have done instead of doing what it can.
INTERNET = """
You also have search_web, which reaches the open web. The corpus comes first -- it is
what this system is for, and it is the only material whose licence and provenance are
known. Use search_web to fill a gap the 34 cultures do not cover, to add context around
what the corpus gave you, or to check a name or transliteration that looks wrong.
Web material is less reliable than the corpus and belongs to nobody here: cite the url,
never credit a sky culture for something that came from the web, and where the two
disagree say so plainly -- "the corpus has X, this source has Y" -- rather than quietly
preferring one.
""".strip()


IMAGES = """
You can put pictures on the screen. show_constellation_image displays the corpus' own
illustration of one figure beside your answer, by the id find_constellation and
compare_across_cultures return. Calling it *is* showing it: you do not embed, link or
paste anything, and you never write out a file name -- if a picture belongs in the
answer, call the tool.

Keep it to one or two in an answer, for figures the answer is really about. A picture
beside a paragraph helps; six pictures bury the paragraph. Only about 300 of the 1529
figures have artwork, and some cultures license their text but not their pictures -- when
one comes back without an image, say so in a few words and move on.

Write the text so it stands on its own: name the culture and what it calls the figure in
prose, and do not refer to a picture with "as you can see above". The picture illustrates
the answer; it is not part of it.
""".strip()


def _language(ctx: RunContext[Deps]) -> str:
    """The one thing the model cannot infer reliably from the prompt alone.

    A question can arrive in English about a Chinese constellation, or in Russian using
    a transliterated name. `lang` is the language the *answer* and the *names* should be
    in, so it is stated rather than guessed.
    """
    return (
        f"The user's language is {ctx.deps.lang!r}. Pass lang={ctx.deps.lang!r} to "
        f"every tool call so names come back in it, and answer in it."
    )


# ─────────────────────────────── the six tool wrappers ───────────────────────────────
#
# One wrapper per tool, each forwarding to `skylore.tools` and nothing else. Parameter
# names must match the schema property names -- that is what carries the descriptions
# across -- and `tests/test_agent.py` asserts it for all six.
#
# **They are `async` for one concrete reason: sqlite connections are bound to the thread
# that opened them.** PydanticAI runs a *sync* tool function in a worker thread, which
# made every call raise `SQLite objects created in a thread can only be used in that same
# thread`. An async function runs in the event loop instead -- the same thread that
# opened the connection -- which fixes it without `check_same_thread=False`, and that
# matters: disabling the check would hand one connection to several threads at once the
# moment a model emits parallel tool calls.
#
# The bodies block the loop, and that is deliberate: a run answers one question, the
# queries are milliseconds, and there is nothing else on the loop to starve. Answering
# many questions at once -- which the evaluation will want -- belongs one level up, as
# one `ask()` per thread with its own connection, not as concurrency inside a run.

async def find_cultures(ctx: RunContext[Deps], query: str | None = None,
                        region: str | None = None, lang: str = "en") -> dict[str, Any]:
    return tools.find_cultures(ctx.deps.connection, query=query, region=region,
                               lang=lang)


async def get_culture_article(ctx: RunContext[Deps], culture: str, lang: str = "en",
                              section: str | None = None) -> dict[str, Any]:
    return tools.get_culture_article(ctx.deps.connection, culture=culture, lang=lang,
                                     section=section)


async def lookup_star(ctx: RunContext[Deps], query: str, lang: str = "en",
                      limit: int = 10) -> dict[str, Any]:
    return tools.lookup_star(ctx.deps.connection, query=query, lang=lang, limit=limit)


async def find_constellation(ctx: RunContext[Deps], query: str, lang: str = "en",
                             culture: str | None = None,
                             limit: int = 10) -> dict[str, Any]:
    return tools.find_constellation(ctx.deps.connection, query=query, lang=lang,
                                    culture=culture, limit=limit)


async def compare_across_cultures(ctx: RunContext[Deps],
                                  constellation: str | None = None,
                                  hips: list[int] | None = None, lang: str = "en",
                                  limit: int = 20) -> dict[str, Any]:
    return tools.compare_across_cultures(ctx.deps.connection,
                                         constellation=constellation, hips=hips,
                                         lang=lang, limit=limit)


async def search_lore(ctx: RunContext[Deps], query: str, lang: str = "en",
                      culture: str | None = None,
                      limit: int = retrieval.TOP_K) -> dict[str, Any]:
    # The embedder is a deployment fact, not a choice the model should be asked to make:
    # `search_lore` falls back to BM25 alone without one and the tool contract is
    # identical either way.
    return tools.search_lore(ctx.deps.connection, query=query, lang=lang,
                             culture=culture, limit=limit,
                             embedder=ctx.deps.embedder)


async def show_constellation_image(ctx: RunContext[Deps], constellation: str,
                                   lang: str = "en") -> dict[str, Any]:
    """Registered only when the caller has somewhere to put a picture, which today means
    the Streamlit chat. The tool returns a path and the caller renders it: a model cannot
    draw on a terminal, and a tool whose result nobody can display is a tool that spends
    tokens describing files.

    **The limit is enforced here, and it was written as a request first.** The schema
    asks for one or two; measurement said that holds until the user asks for a gallery,
    and then it does not -- "покажи, как они выглядят" about five figures produced six
    calls, two of them with invented ids. So the cap is a refusal with a reason rather
    than a sentence in a description. Only displayed pictures count: an error or a
    licence refusal has shown the user nothing and must not consume the allowance.
    """
    if ctx.deps.images_shown >= IMAGE_LIMIT:
        return {"refused": {
            "reason": f"{IMAGE_LIMIT} pictures have already been shown in this answer, "
                      f"which is the limit",
            "hint": "describe the remaining figures in words; do not call this again"}}

    result = tools.show_constellation_image(ctx.deps.connection,
                                            constellation=constellation, lang=lang)
    if result.get("image"):
        ctx.deps.images_shown += 1
    return result


async def search_web(ctx: RunContext[Deps], query: str,
                     max_results: int = web.MAX_RESULTS) -> dict[str, Any]:
    # Registered only when `INTERNET_SEARCH` is on, so reaching this function at all
    # means someone asked for it. Genuinely async, unlike the corpus tools: this one
    # waits on a network rather than on sqlite.
    return await web.search(query, max_results=max_results)


CORPUS_TOOLS = {
    "find_cultures": find_cultures,
    "get_culture_article": get_culture_article,
    "lookup_star": lookup_star,
    "find_constellation": find_constellation,
    "compare_across_cultures": compare_across_cultures,
    "search_lore": search_lore,
}

WEB_TOOLS = {"search_web": search_web}

IMAGE_TOOLS = {"show_constellation_image": show_constellation_image}

WRAPPERS = CORPUS_TOOLS | WEB_TOOLS | IMAGE_TOOLS

SCHEMAS = {schema["name"]: schema
           for schema in [*tools.SCHEMAS, web.SCHEMA, tools.IMAGE_SCHEMA]}


def _docstring(schema: dict[str, Any]) -> str:
    """Render a schema as the Google-style docstring PydanticAI reads.

    Only the `Args:` half matters here -- the summary is passed to `Tool` explicitly and
    wins -- but it is included so the docstring is also useful to a human reading
    `help()`. Properties without a description are skipped rather than given an empty
    entry, which would not parse; `enum` is spelled out because for `region` the allowed
    values *are* the description.
    """
    lines = [schema["description"], "", "Args:"]
    for name, spec in schema["input_schema"]["properties"].items():
        text = spec.get("description", "")
        if spec.get("enum"):
            values = ", ".join(repr(value) for value in spec["enum"])
            text = f"{text} One of: {values}.".strip()
        if not text:
            continue
        lines.append(f"    {name}: {text}")
    return "\n".join(lines)


def toolset(internet: bool | None = None, images: bool = False) -> list[Tool[Deps]]:
    """The tools as PydanticAI sees them, described by their schemas.

    The six corpus tools always; `search_web` only when `INTERNET_SEARCH` says so, and
    `show_constellation_image` only when the caller says it can display one. Both are
    left out entirely rather than registered and refused -- a tool the model cannot see
    is one it cannot misuse, and a refusal in a tool result reads as an invitation to
    rephrase and try again.

    `images` is an argument and not an environment setting, unlike `internet`. Whether
    the web may be reached is a property of the deployment; whether a picture can be
    shown is a property of the caller, and a terminal cannot show one however the
    environment is configured.
    """
    wanted = (CORPUS_TOOLS
              | (WEB_TOOLS if web.enabled(internet) else {})
              | (IMAGE_TOOLS if images else {}))
    built = []
    for name, function in wanted.items():
        schema = SCHEMAS[name]
        function.__doc__ = _docstring(schema)
        built.append(Tool(
            function,
            takes_ctx=True,
            name=name,
            description=schema["description"],
            docstring_format="google",
        ))
    return built


def provider_name(provider: str | None = None) -> str:
    """Which provider to use. `SKYLORE_PROVIDER` so switching one does not touch code."""
    resolved = (provider or os.environ.get("SKYLORE_PROVIDER") or "openai").lower()
    if resolved not in PROVIDERS:
        raise ValueError(f"unknown provider {resolved!r}, expected one of {PROVIDERS}")
    return resolved


def qualify(name: str, provider: str) -> str:
    """OpenRouter namespaces model ids by vendor, so `gpt-5.4-mini` there is
    `openai/gpt-5.4-mini`. Kept pure and separate from building a client because the
    failure it prevents is a 404 that reads as if the model did not exist -- worth a
    test, and a test should not need an API key."""
    if provider == "openrouter" and "/" not in name:
        return f"openai/{name}"
    return name


def model(name: str | None = None, *, provider: str | None = None) -> Model:
    """Resolve a model, on OpenAI or on OpenRouter.

    The key comes from the environment and is passed through rather than checked here:
    PydanticAI's own error names the variable to set, which is more use than anything
    this layer could add.
    """
    resolved = provider_name(provider)
    name = qualify(model_name(name), resolved)
    if resolved == "openrouter":
        return OpenAIChatModel(name, provider=OpenRouterProvider(
            api_key=os.environ.get("OPENROUTER_API_KEY")))
    return OpenAIChatModel(name, provider=OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY")))


def build(llm: Model | str | None = None, internet: bool | None = None,
          images: bool = False) -> Agent[Deps, str]:
    """The agent. Output is prose: what it says is graded by a judge, not by a schema."""
    online = web.enabled(internet)
    if online:
        # Fail here rather than on the first search. Setting INTERNET_SEARCH is someone
        # saying they want the tool; a missing key should not become a tool that
        # silently returns nothing for a whole run.
        web.api_key()
    instructions = [INSTRUCTIONS, _language]
    # Each added only with its tool. An instruction describing a tool the model cannot
    # see is worse than no instruction: it invites the model to explain what it would
    # have done instead of doing what it can.
    if online:
        instructions.insert(1, INTERNET)
    if images:
        instructions.insert(1, IMAGES)
    return Agent(
        llm if llm is not None else model(),
        deps_type=Deps,
        output_type=str,
        instructions=instructions,
        tools=toolset(internet, images=images),
        name="skylore",
    )


# ───────────────────────────────── running one question ─────────────────────────────

@dataclasses.dataclass(frozen=True)
class Call:
    """One tool call the model made. The arguments are kept as sent, so a trajectory can
    be judged on what the model actually asked for -- including whether it carried the
    language through."""

    tool: str
    arguments: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Answer:
    question: str
    text: str
    calls: tuple[Call, ...]
    input_tokens: int
    output_tokens: int
    cost: float | None

    @property
    def tools_used(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(call.tool for call in self.calls))


def trajectory(messages: list[Any]) -> tuple[Call, ...]:
    """The tool calls in a finished run, in order.

    Arguments arrive as a JSON string or as a dict depending on the provider, so both
    are accepted; a string that will not parse is kept verbatim rather than dropped,
    because a malformed call is exactly the kind of thing a trajectory judge should see.
    """
    calls = []
    for message in messages:
        for part in getattr(message, "parts", ()):
            if not isinstance(part, ToolCallPart):
                continue
            arguments = part.args
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"__unparsed__": arguments}
            calls.append(Call(part.tool_name, arguments or {}))
    return tuple(calls)


async def ask_async(question: str, *, lang: str | None = None,
                    llm: Model | str | None = None,
                    embedder: retrieval.Embedder | None = None,
                    internet: bool | None = None,
                    connection: sqlite3.Connection | None = None,
                    images: bool = False,
                    usage_limits: UsageLimits | None = None) -> Answer:
    """Answer one question, and report how it was answered.

    The trajectory and the usage come back beside the text because the agent is going to
    be evaluated on all three: the gold set already separates `question` from `probe`,
    so an agent run can be scored on whether it reached the right material from a
    human-shaped question rather than a hand-written tool call.

    **This is the async form because the evaluation runs many questions at once.** The
    note above the wrappers says concurrency belongs one level up, and it does -- but as
    coroutines on one loop, not as threads: a sqlite connection belongs to the thread
    that opened it, and a connection opened inside this coroutine stays on that thread
    for the whole run. Each caller gets its own by leaving `connection` unset.

    `usage_limits` is what makes a sweep affordable to run at all. `cost_limit` bounds
    one question, so a model that decides to read every article cannot spend the budget
    meant for sixty-eight of them -- the ceiling is enforced by the client, not hoped for.
    """
    owned = connection is None
    connection = connection or tools.connect()
    try:
        agent = build(llm, internet=internet, images=images)
        result = await agent.run(
            question, deps=Deps(connection, lang=language(lang), embedder=embedder),
            usage_limits=usage_limits)
        usage = result.usage
        # `RunUsage.cost` is a `Decimal` of dollars, already priced by genai-prices, and
        # `None` when the price table does not know the model on that provider -- which
        # is not a failed run, so it reports as None rather than as zero. It was read as
        # `usage.cost.total_price` here until the evaluation needed the number and found
        # every run reporting "unpriced": the attribute error was swallowed by a `try`,
        # so the bug was invisible for exactly as long as nobody used the field.
        cost = float(usage.cost) if usage.cost is not None else None
        return Answer(
            question=question,
            text=result.output,
            calls=trajectory(result.all_messages()),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost=cost,
        )
    finally:
        if owned:
            connection.close()


def ask(question: str, *, lang: str | None = None, llm: Model | str | None = None,
        embedder: retrieval.Embedder | None = None,
        internet: bool | None = None,
        connection: sqlite3.Connection | None = None, images: bool = False,
        usage_limits: UsageLimits | None = None) -> Answer:
    """`ask_async` for one question from synchronous code, which is what the CLI is."""
    return asyncio.run(ask_async(question, lang=lang, llm=llm, embedder=embedder,
                                 internet=internet, connection=connection,
                                 images=images, usage_limits=usage_limits))


# ────────────────────────────────────── cli ──────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Ask the agent one question.

        python -m skylore.agent "who else sees a figure in the Pleiades?"
        python -m skylore.agent "кто дал имена звёздам Ориона?" --lang ru --trace
        python -m skylore.agent "is this transliteration right?" --internet

    Reads `.env` at the project root, and lets anything already exported win over it.
    Needs OPENAI_API_KEY, or OPENROUTER_API_KEY with SKYLORE_PROVIDER=openrouter.
    SKYLORE_MODEL and SKYLORE_LANG set the defaults the flags below override. Web
    search is off unless INTERNET_SEARCH=true or --internet, and needs TAVILY_API_KEY.
    """
    import argparse

    load_env()

    parser = argparse.ArgumentParser(description=main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--lang", help=f"the user's language, e.g. ru; "
                                       f"SKYLORE_LANG, else {LANG}")
    parser.add_argument("--model", help=f"SKYLORE_MODEL, else {MODEL}")
    parser.add_argument("--provider", choices=PROVIDERS,
                        help="overrides SKYLORE_PROVIDER")
    parser.add_argument("--embed", metavar="MODEL",
                        help="add the dense half of search_lore, e.g. bge-m3; requires "
                             "the `embed` extra and built vectors")
    parser.add_argument("--internet", action="store_true",
                        help="add search_web for this run; off unless INTERNET_SEARCH "
                             "says otherwise, and needs TAVILY_API_KEY")
    parser.add_argument("--trace", action="store_true",
                        help="print the tool calls and the token cost")
    args = parser.parse_args(argv)

    embedder = None
    if args.embed:
        from ..query import embed as embedding
        embedder = embedding.OnnxEmbedder(embedding.MODELS[args.embed])

    # `--internet` turns it on; without the flag the environment decides, so the default
    # stays off.
    answer = ask(args.question, lang=args.lang, embedder=embedder,
                 internet=True if args.internet else None,
                 llm=model(args.model, provider=args.provider))
    print(answer.text)

    if args.trace:
        print("\n--- trajectory ---")
        for call in answer.calls:
            print(f"  {call.tool}({json.dumps(call.arguments, ensure_ascii=False)})")
        cost = f"${answer.cost:.4f}" if answer.cost is not None else "unpriced"
        print(f"  {answer.input_tokens} in / {answer.output_tokens} out, {cost}")
    return 0

