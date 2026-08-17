# SkyLoreScriber

**A RAG agent over the world's sky cultures.** Ask *"who else sees a figure in the
Pleiades?"* or *"кто дал имена звёздам Ориона?"* and get an answer built from a corpus of
34 cultures — cited, attributed, and in your own language.

## The problem

The same stars have been named by every culture that ever looked up, and the record of it
is scattered across formats nobody can query. [Stellarium's sky-culture
corpus](https://github.com/Stellarium/stellarium-skycultures) is the best public
collection there is — 34 cultures, 1529 constellations, 3735 stars, 14731 names, 1304
article sections — and it ships as markdown articles, `constellations.json` files and
gettext `po` catalogues in four languages. Answering *"which cultures use a star to mark
the rainy season"* means reading all of it.

Three things make this a poor fit for an LLM alone, and the reason this project is a RAG
system rather than a prompt:

1. **The knowledge is not in the model.** Ask any model which cultures draw a figure
   through Alnilam and it will confidently name five, most of them wrong. The corpus knows
   the answer is 25.
2. **Names do not survive translation.** Asked in Chinese about 毕宿, an answer must say
   毕宿 — not translate it to "Net", search, and translate "Net" back into a different
   word. See [Languages](#languages).
3. **The material is licensed.** Five cultures forbid derivatives, one licenses its prose
   but not its illustrations. Attribution is a condition, not a courtesy, so it is
   enforced in code rather than requested in a prompt.

What it does: six retrieval tools over a SQLite corpus, a PydanticAI agent that chooses
between them, an offline gold-set evaluation of the retrieval, an LLM-as-a-judge
evaluation of the prose, a Streamlit chat, and a Grafana dashboard over every run.

---

## Quick start

Eight steps, ~10 minutes, most of it downloads. Everything below is copy-pasteable from a
fresh clone.

### 0. Prerequisites

| | |
|---|---|
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | installs Python 3.14 itself, so nothing else is needed |
| Docker + Docker Compose | for the chat, Postgres and Grafana in step 5 |
| An OpenAI **or** OpenRouter API key | only for the agent — steps 1–4 and 6 need no key and no network |

### 1. Clone, with the corpus

```bash
git clone https://github.com/<you>/SkyLoreScriber.git
cd SkyLoreScriber
git submodule update --init          # the dataset, pinned at the commit it was built from
```

The data is a git submodule, not a download link that rots: `data/data_to_ingest/stellarium-skycultures`
is pinned at an exact upstream commit, so the corpus everyone builds is the corpus these
numbers were measured on. It is ~170 MB and takes a minute.

### 2. Create your `.env`

**Do this before anything else that needs a key.**

```bash
cp .env.example .env
```

Then open `.env` and set **one** provider key:

```bash
OPENAI_API_KEY=sk-...                # the default provider
# or:
OPENROUTER_API_KEY=sk-or-...         # and set SKYLORE_PROVIDER=openrouter
```

Every other line in `.env.example` already holds the value the code falls back to, so an
otherwise untouched copy is a working configuration. `TAVILY_API_KEY` is optional and
enables web search; leaving it empty is a supported state, not an error — the tool is
simply not registered.

The same file configures both the CLI and Docker Compose. Anything already exported in
your shell wins over the file, and a command-line flag wins over both.

### 3. Install

```bash
uv sync                                        # the corpus, the six tools, the gold set — zero dependencies
uv sync --extra agent --extra eval --extra monitor   # everything in this guide
```

Dependencies are pinned in the committed `uv.lock`; `uv sync` reproduces the exact
resolution these numbers came from. The extras are separate because the core genuinely has
no dependencies — `python -m skylore.tools` and the retrieval evaluation run on a bare
Python with no key and no network, and that is worth keeping.

### 4. Build the knowledge base

```bash
python -m skylore.ingest             # ~1 second, writes data/corpus.db (7 MB)
python -m unittest discover tests    # 275 tests, ~1.5s, all offline
```

Fully automated, one command, no notebook: `skylore/ingest/` reads the submodule's
markdown, JSON and `po` catalogues and writes a SQLite database with FTS5 indexes. It is
idempotent — run it again after `git submodule update` and it rebuilds from scratch.
`corpus.db` is deliberately not committed, because it is reproducible from data that is.

### 5. Run the chat + monitoring stack

```bash
docker compose up --build
```

One command brings up all three services:

| | |
|---|---|
| **http://localhost:8501** | the chat — ask questions, 👍/👎 the answers (`APP_PORT` moves it) |
| **http://localhost:3000** | Grafana, dashboard already provisioned (`admin` / `admin`) |
| Postgres | where every run and every verdict is written |

`data/corpus.db` is mounted read-only from the host, which is why step 4 comes first.

**Prefer a terminal?** The same agent, no Docker:

```bash
python -m skylore.agent "who else sees a figure in the Pleiades?" --trace
python -m skylore.agent "кто дал имена звёздам Ориона?" --lang ru
```

`--trace` prints every tool call and what the run cost.

### 6. Evaluate the retrieval

Free, offline, ~2 seconds, no key required:

```bash
python -m scripts.evaluate --validate     # every gold expectation exists in the corpus
python -m scripts.evaluate                # → 65/68 (96%)
```

See [Retrieval evaluation](#retrieval-evaluation) for the comparison of approaches.

### 7. Evaluate the RAG answers (LLM-as-a-judge)

Costs money, so it tells you how much *before* spending it:

```bash
python -m scripts.evaluate_agent --dry-run    # the selection, the ceiling, one full rubric
python -m scripts.evaluate_agent              # 9 questions inside a $0.50 ceiling
```

The dry run prints exactly this, and spends nothing:

```
9 of 68 questions: cross-01, name-01, lang-01, cat-01, article-01, retr-01, cross-02, name-02, lang-02
ceiling $0.45 = 9 x ($0.03 agent + $0.02 judge), budget $0.50
```

See [RAG evaluation](#rag-evaluation) for what is judged and how to A/B two configurations.

### 8. See the monitoring

With the stack from step 5 running, give the dashboard something to draw:

```bash
python -m skylore.monitor seed --count 200            # synthetic traffic — no model, no cost
python -m skylore.monitor stats --source synthetic    # the same totals, to check the panels against
```

Then open **http://localhost:3000** → *SkyLoreScriber*. Ask a few real questions in the
chat and thumb them; `--source chat` separates real traffic from the synthetic rows.

---

## Where each part lives

| | | |
|---|---|---|
| **RAG flow** | knowledge base + LLM, six tools | [`skylore/tools.py`](skylore/tools.py), [`skylore/agent/loop.py`](skylore/agent/loop.py) |
| **Ingestion** | automated, one command | [`skylore/ingest/`](skylore/ingest/) |
| **Retrieval evaluation** | 68 gold questions, 2 approaches compared | [`scripts/evaluate.py`](scripts/evaluate.py), [`data/eval/gold.json`](data/eval/gold.json) |
| **RAG evaluation** | LLM-as-a-judge, 4 checks per question | [`scripts/evaluate_agent.py`](scripts/evaluate_agent.py) |
| **Interface** | Streamlit chat, plus a CLI | [`skylore/monitor/app.py`](skylore/monitor/app.py) |
| **Monitoring** | 13 Grafana panels + user feedback | [`skylore/monitor/`](skylore/monitor/), [`docker/grafana/`](docker/grafana/) |
| **Containerization** | all services in one compose file | [`docker-compose.yaml`](docker-compose.yaml), [`Dockerfile`](Dockerfile) |
| **Hybrid search** | BM25 + dense, fused with RRF | [`skylore/query/retrieval.py`](skylore/query/retrieval.py) |

---

## The six tools

`python -m skylore.tools --schemas` prints what the model sees. Any tool runs from the
shell:

```bash
python -m skylore.tools find_constellation '{"query": "毕宿", "lang": "zh-Hans"}'
python -m skylore.tools compare_across_cultures '{"constellation": "CON western Ori"}'
```

| tool | answers |
|---|---|
| `find_cultures` | the whole 34-row catalogue — a directory to read, not a search |
| `get_culture_article` | one culture's article, whole, with `[#N]` citations resolved |
| `lookup_star` | a star by HIP, international name, Bayer designation, native name or meaning |
| `find_constellation` | fuzzy name match across every culture and language, with its prose |
| `compare_across_cultures` | who else draws a figure through the same stars, and what they saw |
| `search_lore` | prose by meaning, for questions naming nothing in the database |

The tools cut the corpus by *intent* rather than by mechanism, which is why there are six
and not one `search(query)`. All six run with no dependencies. Only `search_lore` improves
with the embedder, and it falls back to BM25 alone without one.

---

## Retrieval evaluation

`data/eval/gold.json` holds **68 questions**, each recording which culture, star,
constellation or section must appear in the result, and *why* it is in the set. Scoring is
recall of required entities — deterministic, free, and offline.

```bash
python -m scripts.evaluate --validate        # rejects a question the corpus cannot answer
python -m scripts.evaluate                   # BM25 alone
python -m scripts.evaluate --embed bge-m3    # BM25 + dense, fused
python -m scripts.evaluate --verbose         # show every question
```

`--validate` has to pass before the score means anything: a gold question expecting a
culture or HIP that does not exist fails forever and teaches nothing.

### Two approaches, measured

`search_lore` is where the choice lives — the other five tools look things up by key.

| approach | retrieval questions | whole gold set |
|---|---|---|
| **BM25 only** (`sections_fts`, no model, no dependencies) | 7/10 | 65/68 (96%) |
| **Hybrid: BM25 + dense, RRF-fused** (`--embed bge-m3`) | 8/10 | **66/68 (97%)** |

The hybrid is what ships, and the reason is specific rather than general: `unicode61` does
not segment CJK, so a run of ideographs becomes a single token and BM25 is simply blind to
Chinese queries. Dense retrieval fixes that case outright. The two rankings are combined
with **reciprocal rank fusion** (`skylore/query/retrieval.py`), with `RRF_K = 60` from the
original paper — measured at 10 and 60, identical scores, so it keeps the published value
rather than a tuned one.

BM25 is kept as a first-class path, not a fallback: it costs nothing, needs no model, and
buys 65 of the 68 points. The dense half is an opt-in extra that adds ~120 MB of runtime
and a 2.3 GB model to win one question.

```bash
uv sync --extra embed
python -m skylore.query.embed --model bge-m3 --langs en    # ~9 min on CPU, once
python -m scripts.evaluate --embed bge-m3
```

`TOP_K = 12` was also measured rather than picked: 8 scored 6/10 and 12 scored 8/10, at
~4.5k tokens rising to ~6.6k. These questions ask which *cultures* do something, and eight
slots under a per-culture cap cannot hold enough distinct ones.

---

## RAG evaluation

The retrieval score says nothing about the prose the agent writes on top, and the prose is
where this project makes most of its promises. `scripts.evaluate_agent` is an
[LLM-as-a-judge](https://ai.pydantic.dev/evals/) run built on `pydantic-evals`.

```bash
uv sync --extra agent --extra eval
python -m scripts.evaluate_agent --dry-run   # the selection, the ceiling, one rubric
python -m scripts.evaluate_agent             # 9 questions inside a $0.50 ceiling
python -m scripts.evaluate_agent --skip 9    # the next nine, without paying twice
python -m scripts.evaluate_agent --ids lang-01 --verbose --record
```

**Four checks per question.** Two are judged by a model, two are read off the trajectory:

| check | source | what it decides |
|---|---|---|
| `grounding` | judge | the gold set's required entities appear in the answer |
| `prose_contract` | judge | attribution credited, a fallback declared as a fallback, a native name given as a name |
| `language_carried` | trajectory, exact | every corpus call asked for the user's language |
| `used_the_corpus` | trajectory, exact | a corpus tool was called before answering |

The last two are deliberately *not* judged. Paying a model to answer what `==` answers is
slower, noisier, dearer, and it lets a judge's mood move a number that has an exact value.

### Comparing configurations

The same gold questions, the same rubrics, one flag apart — which is how any of the
choices above were settled:

```bash
python -m scripts.evaluate_agent --ids lang-01 lang-02 cross-01 --model gpt-5.4-mini
python -m scripts.evaluate_agent --ids lang-01 lang-02 cross-01 --model gpt-5.2
```

`--model` and `--provider` vary the model under test, `--judge` varies the grader. A
judged number means nothing without the model that produced it, so every run prints both
models and the date beside the score. Add `--record` and both runs land in Postgres, where
the Grafana panel *Judge, pass rate by rubric* compares them for you.

The judge is deliberately **not** in the request path: it costs more per question than
answering does, so it grades the gold set rather than live traffic. What runs on every
live question is `skylore/agent/checks.py` — exact, instant, free.

**What the judge cannot catch, stated plainly:** it sees the question and the answer, never
the tool payloads, because those payloads are 855k characters across the gold set and would
multiply the price of every case by its largest term. So it grades form and coverage, not
faithfulness to the sections actually returned. A confident invention in well-formed prose
passes here. Catching that needs the payloads in context and costs roughly an order of
magnitude more — worth doing deliberately, as its own run, not silently on every sweep.

### The bill is bounded before the run

`SKYLORE_EVAL_CAP` bounds one question *inside the client* as
`UsageLimits(cost_limit=...)`, and `SKYLORE_EVAL_BUDGET` decides how many questions that
buys. Cases run = budget / (cap + judge allowance). A model that decides to read every
article stops instead of spending the sweep's budget on one question. Both numbers are
printed before the first request; the actual spend is printed next to the ceiling, because
the gap is the only honest way to size the next sweep.

---

## Languages

The corpus is materialised in **four languages** — English, Russian, Spanish and Chinese
(`en`, `ru`, `es`, `zh_CN`) — and the Chinese variants `zh`, `zh_TW` and `zh_HK` resolve
through declared chains rather than by truncating a tag.

### How to set it

| where | how |
|---|---|
| default for everything | `SKYLORE_LANG=ru` in `.env` |
| one CLI question | `python -m skylore.agent "…" --lang ru` |
| the chat | the **Ask in** selector in the sidebar (`en`, `ru`, `es`, `zh-Hans`) |

Tags are accepted in either BCP 47 or gettext form — `zh-Hans`, `zh_CN`, `ru-RU`, `ru` all
resolve. The flag beats the environment, and the environment beats the default (`en`).

### Why it is three questions and not one

`skylore/lang.py` resolves language **per field**, because "the current language" is three
different orderings of the same four languages:

- **Searching** is wide. A Russian query should match the Russian text directly — that is
  the entire reason four languages are materialised — but it must still reach a Bayer
  designation or an HIP number that only ever appears in the English rows.
- **Prose** comes back in **English**, the source language. Upstream already rendered
  English into Russian, and the answering model translates for the user anyway; serving it
  a corpus translation only adds a second lossy hop.
- **Names** use the **opposite** order: the user's language first. A name is not prose to
  be rendered, it is a term the corpus already holds. Asked in Chinese about 毕宿, an answer
  must say 毕宿 — not "Net", and not whatever "Net" becomes on the way back.

Native names sit outside all of it: `names.kind IN ('native', 'pronounce')` is stored with
`lang IS NULL`, enforced by a schema CHECK, and passed through untouched. A Chinese speaker
asking about a Norse figure wants the Chinese gloss with the Norse name beside it.

Every resolved value carries the language it *actually* came from, because Russian gloss
coverage on stars is 4% and a bare string would pass English off as Russian almost every
time.

And the model has to carry the language into its own tool calls, which is not a promise —
it is the `language_carried` check, measured on every single run.

---

## Monitoring

Evaluation asks how the system does on questions we chose. Monitoring asks what it does on
questions we did not: how long, at what cost, through which tools, and whether anyone was
satisfied.

```bash
docker compose up --build                              # chat + Postgres + Grafana
python -m skylore.monitor seed --count 200             # synthetic traffic, no model, no cost
python -m skylore.monitor stats --source chat          # the same numbers, to check the panels
python -m scripts.evaluate_agent --ids lang-01 --record # a judged sweep, recorded
```

**User feedback** is 👍/👎 under every answer in the chat, written to `verdicts` with
`source='user'`. **The dashboard** is provisioned from `docker/grafana/` — a fresh clone
comes up with the panels already there, rather than with a blank Grafana and a page of
instructions. **13 panels**, in two rows:

*Live traffic* — Runs · Cost · Latency p95 · User feedback (pie) · Trajectory checks pass
rate · Runs over time · Latency · Cost · Tokens · Tools called · Recent runs

*Evaluation* — Judge pass rate by rubric · What the judge failed, and why

Two tables behind them. `runs` is one row per **run of the agent** — question, answer,
model, tokens, cost, seconds, and the trajectory, which is what turns "the answer was
wrong" into "it asked the wrong tool". `verdicts` is what was made of a run, from three
sources kept in one table because they answer the same question:

| source | what | when | cost |
|---|---|---|---|
| `trajectory` | `language_carried`, `used_the_corpus` — read off the tool calls | every run | none |
| `user` | 👍 / 👎 in the chat | when someone bothers | none |
| `judge` | `grounding`, `prose_contract` — a model grading the prose | `evaluate_agent --record` | more than the answer |

Postgres rather than the SQLite this project uses everywhere else, for one reason that is
not about the data: Grafana reads Postgres natively and SQLite only through a third-party
plugin.

---

## The agent

A [PydanticAI](https://ai.pydantic.dev) agent over the six tools — the one part that needs
a network and a key, which is why it is its own extra.

```bash
uv sync --extra agent
python -m skylore.agent "who else sees a figure in the Pleiades?" --trace
```

Defaults to `gpt-5.4-mini`; `--model` and `--provider` override, and on OpenRouter a bare
model id is qualified to `openai/…` rather than 404ing as if it did not exist.

The tool descriptions the model reads are `tools.SCHEMAS` itself, rendered into the wrapper
docstrings rather than paraphrased beside them — those descriptions are what makes the
tools cut by intent, so there is one copy of them. `tests/test_agent.py` runs the whole
layer offline against PydanticAI's stub models and sweeps every wrapper signature against
its schema, because a renamed parameter drops its description with no error anywhere.

One trap worth recording: the wrappers are `async` because a *sync* tool function runs in a
worker thread, and a SQLite connection may only be used in the thread that opened it. The
alternative — `check_same_thread=False` — would hand one connection to several threads the
moment a model emits parallel tool calls.

### Web search, off by default

A seventh tool, `search_web` (Tavily), for what the corpus does not cover or may have got
wrong — it is 34 cultures out of many, its coverage is uneven by language, and an upstream
mis-transliteration stays mis-transliterated here.

```bash
TAVILY_API_KEY=...                            # in .env
python -m skylore.agent "…" --internet        # or INTERNET_SEARCH=true, or the chat's toggle
```

Off unless asked, and when off the tool is **not registered at all** rather than registered
and refused: a tool the model cannot see is one it cannot misuse. Results come back marked
`external: true` with a note that they carry no culture's attribution — in the payload, not
only in the prompt, because an instruction can be forgotten and a field has to be read
past. The rule is not "corpus only" but "corpus first, and never silently replaced": where
the web and the corpus disagree, the answer has to give both and say which said what.

In the chat the toggle is disabled when the key is absent — a switch whose only effect is
an exception is not a switch — and the run records that the web was *offered*, which
`tools` cannot tell you: an answer where the model had the web and stayed in the corpus is
the interesting one.

### Pictures

`show_constellation_image` is registered whenever there is a screen to draw on, and the
model calls it to put the corpus' illustration beside its answer. Two per answer, enforced
in the wrapper rather than requested in the description — the description asked first, and
measurement said asking holds until someone says "show me all five", which produced six
calls. Artwork is licence-bound in both directions: a culture may licence its prose and not
its pictures (maori does), a single file may be carved out by review, and the tool checks
both plus the file's existence, because upstream declares two illustrations it does not
ship.

---

## Reading the code

The design is in the code, next to what it decides: `skylore/__init__.py` maps the four
layers and the order to read them in, and each module opens with why it is shaped the way
it is. Three things worth knowing first:

**Language is three questions, not one** — see [Languages](#languages) above.

**Licensing is enforced in code, not documented in prose.** Five cultures are excluded for
no-derivatives clauses. `japanese_moon_stations/chart.webp` is admitted on the condition
that the map is never served, and its reference is embedded in that culture's prose in all
four languages — so it is stripped at query time under two independent rules, and a test
sweeps all 34 cultures. Attribution is derived from each response rather than passed in,
because a licence condition a caller has to remember is one that eventually gets forgotten.

**The retrieval unit is the whole section.** ~1020 of them, averaging 728 characters.
Nothing is chunked: at this size, keeping a section intact costs less than splitting a
markdown table or orphaning a paragraph from its heading, and returning eight whole ones
costs a few thousand tokens with no chunk-boundary loss at all.
