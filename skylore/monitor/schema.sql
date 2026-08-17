-- What happened when someone asked, and what anyone thought of the answer.
--
-- Two tables, and the split between them is the point: `runs` is what the system did,
-- `verdicts` is what was made of it. A run is written once and never updated; verdicts
-- arrive afterwards, from three different places, at three different times -- the
-- trajectory checks within milliseconds, a user's thumb within seconds, a judge only
-- when an evaluation is run. Folding a verdict into the run row would mean either
-- updating a row that is already true or waiting to write it until every opinion is in.
--
-- Applied by `store.init()` on every start, which is why every statement is
-- `IF NOT EXISTS`. The alternative -- a hand-run init script that drops and recreates --
-- makes a fresh clone a two-command affair and loses the data the second time somebody
-- runs it by mistake.

-- ─────────────────────────────────── runs ───────────────────────────────────
--
-- One row per *run of the agent*, not per model request. The unit a person experiences
-- is "I asked, it answered": one question can be three requests and four tool calls, and
-- splitting it into request rows would make "what did that answer cost" a join instead
-- of a column.

CREATE TABLE IF NOT EXISTS runs (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Where the question came from. Live traffic and gold-set questions land in the same
    -- table on purpose: "what does a question cost in production" and "what did the last
    -- evaluation cost" are then the same query with a different filter, and a dashboard
    -- can show the two side by side. `synthetic` is `skylore.monitor.synthetic`, so a
    -- demo dashboard can be full without anyone spending anything.
    source        TEXT NOT NULL CHECK (source IN ('chat', 'eval', 'synthetic')),

    question      TEXT NOT NULL,
    lang          TEXT NOT NULL,
    answer        TEXT NOT NULL,

    -- Recorded per run rather than assumed from configuration: the model is a setting
    -- (`SKYLORE_MODEL`), it changes, and a number without the model that produced it
    -- means nothing a week later.
    model         TEXT NOT NULL,
    provider      TEXT NOT NULL,

    -- The trajectory, twice, for two different readers. `tools` is the flat list a panel
    -- aggregates (`SELECT unnest(tools), count(*) ... GROUP BY 1`); `calls` keeps every
    -- call with its arguments, which is what turns "the answer was wrong" into "it asked
    -- for the wrong thing" -- including a malformed call, kept verbatim upstream rather
    -- than dropped.
    tools         TEXT[] NOT NULL,
    calls         JSONB  NOT NULL,

    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,

    -- Nullable, and that is the contract: genai-prices does not know every model on every
    -- provider, and an unpriced run is not a free one. A dashboard that summed NULL as
    -- zero would quietly under-report the bill.
    cost          DOUBLE PRECISION,

    seconds       DOUBLE PRECISION NOT NULL,

    -- The gold question's id when `source = 'eval'`, NULL otherwise. It is what lets a
    -- judged verdict be traced back to the expectation it was judged against.
    case_id       TEXT,

    -- Whether `search_web` was on the table for this run. Not derivable from `tools`:
    -- that records the tools the model *called*, and the interesting run is the one where
    -- the web was offered and refused. It also explains a cost and a latency that would
    -- otherwise look like noise, and it is the flag that says an answer may contain
    -- material the corpus never licensed.
    internet      BOOLEAN NOT NULL DEFAULT false
);

-- Columns added after the first database was created. `CREATE TABLE IF NOT EXISTS` does
-- nothing to a table that already exists, so a new column in the block above would never
-- reach a store that has been running -- it would exist in the file, in the tests and in
-- nobody's database. This is the whole migration story the project needs at this size:
-- one idempotent statement per column, kept in order, never removed.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS internet BOOLEAN NOT NULL DEFAULT false;

-- Every panel filters on a time range, and the table only ever grows.
CREATE INDEX IF NOT EXISTS ix_runs_ts     ON runs (ts DESC);
CREATE INDEX IF NOT EXISTS ix_runs_source ON runs (source, ts DESC);

-- ───────────────────────────────── verdicts ─────────────────────────────────
--
-- One table for three kinds of opinion, distinguished by `source`, because they answer
-- the same question -- was this answer any good -- and separating them would mean a
-- dashboard querying three tables to show one bar chart.
--
--   trajectory  read off the tool calls, on every run, for nothing. Exact.
--   judge       a model grading the prose. Only from `scripts.evaluate_agent`: the
--               judge costs more per question than the answer does, so it grades the
--               gold set rather than every live question.
--   user        a thumb in the Streamlit app. The only signal here that knows what the
--               person actually wanted.

CREATE TABLE IF NOT EXISTS verdicts (
    id        BIGSERIAL PRIMARY KEY,
    run_id    BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ts        TIMESTAMPTZ NOT NULL DEFAULT now(),

    source    TEXT NOT NULL CHECK (source IN ('trajectory', 'judge', 'user')),

    -- Which check: `language_carried`, `used_the_corpus`, `grounding`, `prose_contract`,
    -- `thumbs`. Not constrained, because the set of evaluators is expected to grow and a
    -- CHECK here would make adding one a migration.
    evaluator TEXT NOT NULL,

    -- `passed` for a check, `score` for a thumb (+1/-1). Neither is filled for both, so
    -- both are nullable rather than forced into one column that means two things.
    passed    BOOLEAN,
    score     INTEGER,

    -- Why. A pass rate says something is wrong; only this says what.
    reason    TEXT,

    -- The judging model, NULL for the checks and the thumbs. Same argument as `runs.model`
    -- -- a judged number is meaningless without it.
    model     TEXT
);

CREATE INDEX IF NOT EXISTS ix_verdicts_run    ON verdicts (run_id);
CREATE INDEX IF NOT EXISTS ix_verdicts_source ON verdicts (source, ts DESC);
