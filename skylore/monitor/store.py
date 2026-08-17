"""The Postgres side: where the rows go and how they come back.

Postgres rather than the sqlite this project uses everywhere else, for one reason that
is not about the data: Grafana reads Postgres natively and sqlite only through a
third-party plugin. The corpus stays in sqlite -- it is read-only, reproducible and
shipped -- and this database is neither: it is append-only, it outlives any particular
checkout, and something other than Python has to be able to read it.

Written with `psycopg` and no ORM. There are four statements in the whole module and the
schema is fifty lines; an ORM here would add a dependency, a mapping layer and a
migration story to save nothing.

Connections are opened per call rather than pooled. A run costs seconds of model time
and one insert of a few hundred microseconds, so a pool would optimise the part that is
already free -- and a pooled connection held across a Streamlit rerun is a class of bug
worth not having.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import psycopg

from ..paths import MONITOR_SCHEMA


def dsn() -> str:
    """Where the database is, from the environment.

    The defaults are the ones `docker-compose.yaml` sets, so `docker compose up` needs no
    configuration at all, and a developer running the app against a compose-managed
    Postgres needs only `POSTGRES_HOST=localhost`. There is no password default worth
    defending here: this is a local monitoring store, and pretending otherwise by
    demanding one would only teach the habit of putting a real password in `.env`.
    """
    return (
        f"host={os.environ.get('POSTGRES_HOST', 'localhost')} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ.get('POSTGRES_DB', 'skylore')} "
        f"user={os.environ.get('POSTGRES_USER', 'skylore')} "
        f"password={os.environ.get('POSTGRES_PASSWORD', 'skylore')}"
    )


def connect() -> psycopg.Connection:
    return psycopg.connect(dsn())


def init() -> None:
    """Apply the schema. Idempotent, so it runs on every start.

    The reference implementation this follows keeps a `db_init.py` that drops and
    recreates, run by hand before the app -- which makes a fresh clone a two-command
    affair and deletes the history the second time somebody runs it out of habit. Every
    statement in `schema.sql` is `IF NOT EXISTS`, so the app can simply call this at
    startup and the question never comes up.
    """
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(MONITOR_SCHEMA.read_text(encoding="utf-8"))


def reset() -> None:
    """Drop both tables. Only for `python -m skylore.monitor init --reset`.

    Kept apart from `init` and never called by the app: a schema change during
    development is a real need, and losing a week of traffic to a startup path that
    dropped tables is a real accident.
    """
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS verdicts, runs CASCADE")


# ─────────────────────────────────── writing ───────────────────────────────────

def save_run(*, source: str, question: str, lang: str, answer: str, model: str,
             provider: str, tools: Sequence[str], calls: Sequence[dict[str, Any]],
             input_tokens: int, output_tokens: int, cost: float | None,
             seconds: float, case_id: str | None = None, internet: bool = False,
             ts: datetime | None = None) -> int:
    """Write one finished run and return its id.

    Keyword-only, all of it. The row has fourteen columns, six of them numbers, and a
    positional call that swaps `input_tokens` and `output_tokens` would produce a
    perfectly valid row with a wrong bill in it.

    `ts` defaults to the database's `now()` and exists for exactly one caller:
    `synthetic`, which has to spread its rows over past hours. Two hundred rows written
    in one second make every time-series panel a single spike, which is a poor way to
    find out whether the panel works.
    """
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO runs (ts, source, question, lang, answer, model, provider,
                              tools, calls, input_tokens, output_tokens, cost,
                              seconds, case_id, internet)
            VALUES (coalesce(%s, now()), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s)
            RETURNING id
            """,
            (ts, source, question, lang, answer, model, provider, list(tools),
             json.dumps(calls, ensure_ascii=False), input_tokens, output_tokens,
             cost, seconds, case_id, internet),
        )
        return cursor.fetchone()[0]


def save_verdict(run_id: int, *, source: str, evaluator: str,
                 passed: bool | None = None, score: int | None = None,
                 reason: str | None = None, model: str | None = None,
                 ts: datetime | None = None) -> None:
    """Write one opinion about one run: a check, a judge's grade or a user's thumb."""
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO verdicts (run_id, ts, source, evaluator, passed, score, reason,
                                  model)
            VALUES (%s, coalesce(%s, now()), %s, %s, %s, %s, %s, %s)
            """,
            (run_id, ts, source, evaluator, passed, score, reason, model),
        )


# ─────────────────────────────────── reading ───────────────────────────────────
#
# Grafana does its own SQL against these tables and needs nothing from here. These two
# exist so the numbers on the dashboard can be checked against the database from a
# terminal -- a dashboard nobody can verify is a dashboard nobody should trust.

@dataclass
class Stats:
    runs: int
    cost: float | None
    seconds: float | None
    input_tokens: float | None
    output_tokens: float | None
    passed: int
    failed: int
    thumbs_up: int
    thumbs_down: int


def stats(source: str | None = None) -> Stats:
    """Totals, optionally for one kind of traffic."""
    where = "WHERE source = %s" if source else ""
    joined = "WHERE r.source = %s" if source else ""
    params = (source,) if source else ()
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT count(*), sum(cost), avg(seconds), avg(input_tokens),
                   avg(output_tokens)
            FROM runs {where}
            """, params)
        runs, cost, seconds, input_tokens, output_tokens = cursor.fetchone()

        # Verdicts are filtered through the runs they belong to, so `--source chat` means
        # the same thing in both halves of the answer.
        cursor.execute(
            f"""
            SELECT
                count(*) FILTER (WHERE v.passed IS TRUE),
                count(*) FILTER (WHERE v.passed IS FALSE),
                count(*) FILTER (WHERE v.score > 0),
                count(*) FILTER (WHERE v.score < 0)
            FROM verdicts v JOIN runs r ON r.id = v.run_id
            {joined}
            """, params)
        passed, failed, up, down = cursor.fetchone()

    return Stats(runs=runs, cost=cost, seconds=seconds, input_tokens=input_tokens,
                 output_tokens=output_tokens, passed=passed, failed=failed,
                 thumbs_up=up, thumbs_down=down)


def recent(limit: int = 10) -> list[dict[str, Any]]:
    """The last few runs, newest first, with the verdicts attached to each."""
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT r.id, r.ts, r.source, r.question, r.lang, r.answer, r.model,
                   r.tools, r.cost, r.seconds, r.internet,
                   coalesce(
                       (SELECT json_agg(json_build_object(
                                   'source', v.source, 'evaluator', v.evaluator,
                                   'passed', v.passed, 'score', v.score,
                                   'reason', v.reason)
                                ORDER BY v.id)
                        FROM verdicts v WHERE v.run_id = r.id),
                       '[]'::json)
            FROM runs r
            ORDER BY r.ts DESC
            LIMIT %s
            """, (limit,))
        columns = ("id", "ts", "source", "question", "lang", "answer", "model",
                   "tools", "cost", "seconds", "internet", "verdicts")
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
