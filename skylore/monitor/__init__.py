"""What happened in production, written down: runs, their trajectories, and verdicts.

Evaluation asks whether the system is any good on questions we chose. This layer asks
what it does on questions we did not choose -- how long, at what cost, through which
tools, and whether anyone was satisfied. The two are not substitutes: a gold set cannot
tell you that the median question now costs three times what it did last week, and a
dashboard cannot tell you that an answer was wrong.

    store       the connection, the schema and the four statements anything here needs.
                `schema.sql` sits beside it for the same reason `ingest/schema.sql` does:
                the shape of a database is code.

    record      one finished `loop.Answer` becomes one row, plus the two verdicts that
                `agent.checks` can settle without asking a model.

    synthetic   plausible rows with no model behind them, so a dashboard can be
                demonstrated, and its panels debugged, without spending anything.

    app         the Streamlit chat. Ask, read, thumb. `python -m streamlit run
                skylore/monitor/app.py`, or `docker compose up`.

                `python -m skylore.monitor init | seed | stats`

**The judge is not here.** It grades prose with a model, which costs more per question
than answering did, so it runs in `scripts.evaluate_agent` against the gold set and
writes its verdicts into the same table with `source='eval'`. What runs on live traffic
is `agent.checks`: exact, instant and free. That split is the whole reason a live answer
costs exactly what the answer costs.

**Nothing below this layer knows it exists.** `skylore.agent` does not import it, and a
run is recorded by the caller that wanted it recorded -- the app, or the evaluation with
`--record`. Monitoring that reaches down into the thing it monitors is how a database
outage becomes an outage.
"""
