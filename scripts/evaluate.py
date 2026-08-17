"""Validate and score the gold evaluation set.

    python -m scripts.evaluate --validate    # every expectation exists in the corpus
    python -m scripts.evaluate               # run the runnable paths and score them
    python -m scripts.evaluate --verbose     # show each question

Two commands because they answer different questions, and the first has to pass before
the second means anything. `--validate` checks the *set* against the corpus: a gold
question expecting a culture, HIP, constellation or heading path that does not exist is
worse than no question at all, because it fails forever and teaches nothing. `--score`
checks the *system* against the set.

`--embed MODEL` adds the dense half to the retrieval path; without it `search_lore`
runs on BM25 alone. Both numbers are worth keeping: BM25 alone scores 7/10 on the
retrieval questions and needs no model at all, and bge-m3 takes that to 8/10 by fixing
the Chinese case outright. Reporting only the better one would hide how much the
dependency actually buys.

Scoring is recall of required entities, not answer quality. Whether an answer reads
well is not measurable here and is not what this set is for.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skylore import tools  # noqa: E402
from skylore.paths import GOLD  # noqa: E402

# `retrieval` is runnable on BM25 alone; the dense half only changes how well it scores,
# not whether it runs. Kept listed so a score always says which halves were in play.
RUNNABLE = {"catalogue", "name", "star", "article", "retrieval"}


@dataclass
class Result:
    question_id: str
    path: str
    passed: bool
    detail: str


# ─────────────────────────────────── validation ───────────────────────────────────

def validate(connection: sqlite3.Connection, gold: dict) -> list[str]:
    """Every entity a question expects must exist. Returns the problems found."""
    problems: list[str] = []

    known_cultures = {row[0] for row in connection.execute("SELECT id FROM cultures")}
    known_hips = {row[0] for row in connection.execute("SELECT hip FROM stars")}
    known_constellations = {row[0] for row in connection.execute(
        "SELECT id FROM constellations")}
    known_sections = {(row[0], row[1]) for row in connection.execute(
        "SELECT culture_id, heading_path FROM sections WHERE lang = 'en'")}

    seen_ids: set[str] = set()
    for question in gold["questions"]:
        qid = question["id"]
        if qid in seen_ids:
            problems.append(f"{qid}: duplicate id")
        seen_ids.add(qid)

        if question["path"] not in gold["paths"]:
            problems.append(f"{qid}: unknown path {question['path']!r}")
        for field in ("question", "probe", "lang", "why"):
            if field not in question:
                problems.append(f"{qid}: missing {field!r}")

        expect = question.get("expect", {})
        if not expect:
            problems.append(f"{qid}: no expectations, so it can never fail")

        for culture in expect.get("cultures", []):
            if culture not in known_cultures:
                problems.append(f"{qid}: no such culture {culture!r}")
        for hip in expect.get("hips", []):
            if hip not in known_hips:
                problems.append(f"{qid}: no such star HIP {hip}")
        for constellation in expect.get("constellations", []):
            if constellation not in known_constellations:
                problems.append(f"{qid}: no such constellation {constellation!r}")
        for culture, heading in expect.get("sections", []):
            if (culture, heading) not in known_sections:
                problems.append(f"{qid}: no such section {culture!r} / {heading!r}")

        # A retrieval question the name path already answers measures nothing about
        # retrieval. Catching that here is the difference between a set that grows and
        # one that grows honestly.
        if question["path"] == "retrieval" and question["probe"].strip():
            from skylore.query import names as name_lookup
            hits = name_lookup.search(connection, question["probe"], limit=1)
            if hits:
                problems.append(
                    f"{qid}: probe matches the name {hits[0].value!r} directly, so it "
                    f"does not test retrieval")

    return problems


# ──────────────────────────────────── scoring ────────────────────────────────────

def _entities(payload: dict) -> tuple[set[str], set[int], set[str]]:
    """Cultures, HIPs and constellation ids anywhere in a tool response."""
    cultures: set[str] = set()
    hips: set[int] = set()
    constellations: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "culture" and isinstance(value, str):
                    cultures.add(value)
                elif key == "hip" and isinstance(value, int):
                    hips.add(value)
                elif key == "id" and isinstance(value, str):
                    (constellations if value.startswith("CON ") else cultures).add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk({k: v for k, v in payload.items() if k != "sources"})
    return cultures, hips, constellations


def _fallback_flags(payload: dict) -> set[bool]:
    flags: set[bool] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            if "value" in node and "lang" in node:
                flags.add(bool(node.get("is_fallback")))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk({k: v for k, v in payload.items() if k != "sources"})
    return flags


def _call(connection: sqlite3.Connection, question: dict,
          embedder=None) -> dict:
    path, probe, language = question["path"], question["probe"], question["lang"]
    if path == "catalogue":
        return tools.find_cultures(connection, region=question.get("region"),
                                   lang=language)
    if path == "name":
        return tools.find_constellation(connection, query=probe, lang=language)
    if path == "star":
        return tools.lookup_star(connection, query=probe, lang=language)
    if path == "article":
        return tools.get_culture_article(connection, culture=probe, lang=language)
    if path == "retrieval":
        return tools.search_lore(connection, query=probe, lang=language,
                                 embedder=embedder)
    raise ValueError(path)


def score(connection: sqlite3.Connection, question: dict,
          embedder=None) -> Result:
    qid, expect = question["id"], question["expect"]
    payload = _call(connection, question, embedder)
    cultures, hips, constellations = _entities(payload)
    missing: list[str] = []

    required_cultures = set(expect.get("cultures", []))
    if not required_cultures <= cultures:
        missing.append(f"cultures {sorted(required_cultures - cultures)}")
    if expect.get("exact_cultures") and cultures != required_cultures:
        missing.append(f"unexpected cultures {sorted(cultures - required_cultures)}")

    if not set(expect.get("hips", [])) <= hips:
        missing.append(f"HIPs {sorted(set(expect['hips']) - hips)}")
    required_constellations = set(expect.get("constellations", []))
    if not required_constellations <= constellations:
        missing.append(
            f"constellations {sorted(required_constellations - constellations)}")

    for culture, heading in expect.get("sections", []):
        headings = {s["heading_path"] for s in payload.get("sections", [])}
        if heading not in headings:
            missing.append(f"section {heading!r}")

    minimum = expect.get("min_cultures_naming")
    if minimum is not None:
        naming = len({n["culture"] for star in payload.get("stars", [])
                      for n in star["named_by"] if n["culture"]})
        if naming < minimum:
            missing.append(f"only {naming} cultures name it, expected >= {minimum}")

    minimum = expect.get("min_cultures_drawing")
    if minimum is not None:
        drawing = len({c["culture"] for star in payload.get("stars", [])
                       for c in star["drawn_into"]})
        if drawing < minimum:
            missing.append(f"only {drawing} cultures draw it, expected >= {minimum}")

    minimum = expect.get("min_cultures")
    if minimum is not None and len(cultures) < minimum:
        missing.append(f"only {len(cultures)} cultures, expected >= {minimum}")

    flags = _fallback_flags(payload)
    if expect.get("expect_fallback") and True not in flags:
        missing.append("expected a fallback to be flagged, none was")
    if expect.get("expect_no_fallback") and flags != {False} and flags:
        missing.append("expected no fallback flag, but one was set")

    return Result(qid, question["path"], not missing, "; ".join(missing) or "ok")


# ────────────────────────────────────── cli ──────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true",
                        help="check the set against the corpus and stop")
    parser.add_argument("--verbose", action="store_true", help="show every question")
    parser.add_argument("--path", help="score only one path")
    parser.add_argument("--embed", metavar="MODEL",
                        help="use a dense model for the retrieval path, e.g. bge-m3; "
                             "requires the `embed` extra and built vectors")
    args = parser.parse_args(argv)

    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    connection = tools.connect()

    problems = validate(connection, gold)
    if problems:
        print(f"{len(problems)} problems in the gold set:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"validated: {len(gold['questions'])} questions, every expectation exists")
    if args.validate:
        return 0

    embedder = None
    if args.embed:
        from skylore.query import embed as embedding, retrieval
        embedder = embedding.OnnxEmbedder(embedding.MODELS[args.embed])
        built = retrieval.embedded_langs(connection, args.embed)
        if not built:
            print(f"no vectors for {args.embed}; run python -m skylore.query.embed --model "
                  f"{args.embed} --langs en")
            return 1
        print(f"dense half: {args.embed} over {built}")

    results: list[Result] = []
    pending = 0
    for question in gold["questions"]:
        if args.path and question["path"] != args.path:
            continue
        if question["path"] not in RUNNABLE:
            pending += 1
            if args.verbose:
                print(f"  PEND {question['id']:12} {question['path']}")
            continue
        result = score(connection, question, embedder)
        results.append(result)
        if args.verbose or not result.passed:
            mark = "pass" if result.passed else "FAIL"
            print(f"  {mark} {result.question_id:12} {result.detail}")

    passed = sum(r.passed for r in results)
    by_path: dict[str, list[Result]] = {}
    for result in results:
        by_path.setdefault(result.path, []).append(result)

    print(f"\nscored {len(results)} runnable, {pending} pending on search_lore")
    for path in sorted(by_path):
        group = by_path[path]
        print(f"  {path:10} {sum(r.passed for r in group)}/{len(group)}")
    print(f"  {'total':10} {passed}/{len(results)}"
          f"  ({passed / len(results):.0%})" if results else "")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
