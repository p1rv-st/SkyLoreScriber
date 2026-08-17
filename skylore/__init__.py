"""Retrieval over the world's sky cultures, for an agent to answer questions with.

The package is four layers, and every import runs one way -- down. Reading it in this
order means never needing a module that has not been introduced yet.

    paths, lang         where the data lives, and which locale a field resolves to.
                        Both halves below need them, which is why they stay here rather
                        than in either.

    ingest/             the corpus as upstream ships it: `po` reads the translation
                        catalogues, `corpus` parses `index.json` and `description.md`,
                        `build` writes them into sqlite against `schema.sql`.
                        `python -m skylore.ingest`

    query/              the corpus as a question sees it: `names` finds a name in any
                        form, `cultures` serves the catalogue and the articles,
                        `retrieval` ranks prose, `embed` builds the dense half of that
                        ranking, `compare` joins a figure across cultures.

    tools               the boundary: the six tools a model is given, their schemas and
                        their JSON. Everything above is what they are built from; nothing
                        below this line knows a model exists.
                        `python -m skylore.tools`

    agent/              the RAG loop over those six, plus `web` -- a seventh tool, off
                        unless asked for, and the only one that leaves the corpus, and
                        `checks`, which reads a finished trajectory without a model.
                        `python -m skylore.agent`

    monitor/            what happened when someone asked: runs, trajectories and
                        verdicts in Postgres, a Streamlit chat over the agent and
                        Grafana over the tables. The only layer that writes anywhere
                        but the corpus, and the only one nothing else imports -- it
                        depends on `agent`, and `agent` must never depend on it.
                        `python -m skylore.monitor` / `docker compose up`

One arrow runs against the grain: `query/cultures.py` and `query/retrieval.py` import
`ingest.corpus` for `cited_refs` and `localise_section`. Those two are text handling, not
ingest, but they live beside the parser that produced the text and splitting them out
would leave a module of two functions. Recorded here rather than hidden, so the next
reader knows it is a decision and not a stray import.
"""
