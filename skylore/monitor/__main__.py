"""The monitoring store from a terminal.

    python -m skylore.monitor init              # create the tables; safe to repeat
    python -m skylore.monitor init --reset      # drop them first. Destroys the history
    python -m skylore.monitor seed --count 200  # synthetic traffic, no model, no cost
    python -m skylore.monitor stats             # what the dashboard should be showing
    python -m skylore.monitor stats --source chat

`stats` exists so that the dashboard can be checked against the database it draws from.
A panel and a query that disagree is a thing worth finding out about from a terminal
rather than from a decision made on a wrong number.

Reads `POSTGRES_*` from the environment, and `.env` if there is one -- the same
resolution `skylore.agent` uses for its keys.
"""

from __future__ import annotations

import argparse

from ..agent.loop import load_env
from . import store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    initialise = commands.add_parser("init", help="create the tables")
    initialise.add_argument("--reset", action="store_true",
                            help="drop them first -- destroys everything recorded")

    seed = commands.add_parser("seed", help="write synthetic traffic")
    seed.add_argument("--count", type=int, default=200)
    seed.add_argument("--hours", type=float, default=6.0,
                     help="spread the rows over this many past hours; default 6, which "
                          "is the dashboard's default range")
    seed.add_argument("--seed", type=int, help="make the run reproducible")

    stats = commands.add_parser("stats", help="totals, to check the dashboard against")
    stats.add_argument("--source", choices=("chat", "eval", "synthetic"),
                       help="only this kind of traffic")

    args = parser.parse_args(argv)
    load_env()

    if args.command == "init":
        if args.reset:
            store.reset()
            print("dropped runs and verdicts")
        store.init()
        print(f"schema applied to {store.dsn().split('password=')[0].strip()}")
        return 0

    if args.command == "seed":
        from . import synthetic

        store.init()
        written = synthetic.seed(count=args.count, hours=args.hours, seed=args.seed)
        print(f"wrote {written} synthetic runs over the last {args.hours:g}h")
        return 0

    totals = store.stats(args.source)
    scope = args.source or "all sources"
    print(f"{totals.runs} runs, {scope}")
    if not totals.runs:
        return 0
    cost = f"${totals.cost:.4f}" if totals.cost is not None else "unpriced"
    print(f"  {cost} total, {totals.seconds:.1f}s and "
          f"{totals.input_tokens:,.0f} in / {totals.output_tokens:,.0f} out a run")
    print(f"  verdicts: {totals.passed} passed, {totals.failed} failed")
    print(f"  users: {totals.thumbs_up} up, {totals.thumbs_down} down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
