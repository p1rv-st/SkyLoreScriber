"""The chat, and the only place a human opinion enters the system.

    docker compose up
    streamlit run skylore/monitor/app.py     # against a Postgres you already have

Deliberately only a chat. The dashboards are Grafana's, provisioned from
`docker/grafana/`, and a second dashboard here would be a second set of queries to keep
in step with the first -- with the added cost that whichever one somebody looked at,
they would have to remember which numbers it was allowed to show.

What it does show is the run it just made: how long, how many tokens, what it cost, and
which tools were called with which arguments. Those are on the page rather than only in
the database because the person best placed to notice that an answer came from the wrong
tool is the one reading the answer.

The thumbs are the point of the whole page. Everything else here is measurable without a
person -- the trajectory checks run on every question, the judge grades the gold set --
but whether the answer was what somebody actually wanted is not derivable from anything
this system knows. That signal only exists if it is asked for.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# `streamlit run` executes this file as a script, not as `skylore.monitor.app`, so there
# is no package to be relative to and the repository root is not on the path. Every other
# entry point in the project is a `python -m`, which needs neither line; this one is the
# exception because Streamlit chose how it starts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from skylore import paths, tools  # noqa: E402
from skylore.agent import loop  # noqa: E402
from skylore.monitor import record, store  # noqa: E402

# Read once per session rather than per rerun: Streamlit re-executes this file top to
# bottom on every interaction, and `.env` does not change between two clicks.
LANGS = ["en", "ru", "es", "zh-Hans"]


@st.cache_resource
def start() -> str:
    """Load the environment and make sure the tables exist. Once per process.

    `store.init()` is idempotent, so calling it at startup removes the hand-run
    initialisation step entirely: a fresh `docker compose up` against an empty database
    comes up working rather than coming up broken in a way that looks like a bug.
    """
    loop.load_env()
    store.init()
    return loop.model_name()


def show_images(answer) -> None:
    """Render whatever the model asked to be shown, in the order it asked.

    The trajectory is the instruction. `Answer` carries the calls but not their results,
    so each one is resolved again here -- a sqlite read against a read-only corpus,
    costing nothing and asking no model. Resolving rather than trusting also means the
    licence check in `tools.show_constellation_image` runs on the display path too: a
    picture is only rendered if the rule that governs it says so at the moment it is
    rendered.
    """
    wanted = [call.arguments.get("constellation") for call in answer.calls
              if call.tool == "show_constellation_image"]
    wanted = list(dict.fromkeys(filter(None, wanted)))
    if not wanted:
        return

    # The wrapper refuses past `IMAGE_LIMIT`, but the trajectory records the calls it
    # refused alongside the ones it served, and this function resolves calls rather than
    # results. Without the same limit here the page would render every figure the model
    # asked for -- and the run that produced this line asked for six.
    connection = tools.connect()
    rendered = 0
    try:
        columns = st.columns(min(len(wanted), loop.IMAGE_LIMIT))
        for constellation in wanted:
            if rendered >= loop.IMAGE_LIMIT:
                break
            result = tools.show_constellation_image(
                connection, constellation=constellation, lang=lang)
            image = result.get("image")
            if not image:
                # Either no artwork or a licence refusal. The model has already been told
                # in prose; nothing goes on the page, because an empty frame saying
                # "withheld" is still an assertion about a picture we may not show.
                continue
            name = result.get("name") or {}
            caption = " — ".join(part for part in (
                name.get("native") or name.get("pronounce"),
                (name.get("meaning") or {}).get("value"),
                result["culture"],
            ) if part)
            columns[rendered % len(columns)].image(
                str(paths.CORPUS_DIR / image["path"]), caption=caption)
            rendered += 1
    finally:
        connection.close()


model = start()

st.set_page_config(page_title="SkyLoreScriber", page_icon="✨")
st.title("SkyLoreScriber")
st.caption(f"34 sky cultures, six tools, {model}. Every question is recorded; "
           f"the dashboard is in Grafana.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Ask in")
    default = loop.language()
    lang = st.selectbox(
        "language", LANGS,
        index=LANGS.index(default) if default in LANGS else 0,
        help="The language the answer and the names come back in. The model is told it "
             "and has to carry it into its own tool calls -- which is what the "
             "`language_carried` check on every run measures.")

    st.divider()
    st.subheader("Beyond the corpus")

    # The toggle is offered only when the key is there. `web.api_key()` raises at build
    # time when it is not -- deliberately, because asked-for-and-broken is worse than off
    # -- and a switch whose only effect is an exception is not a switch.
    has_key = bool(os.environ.get("TAVILY_API_KEY", "").strip())
    internet = st.toggle(
        "Search the web",
        value=False,
        disabled=not has_key,
        help="Adds `search_web` (Tavily) as a seventh tool for this question. The corpus "
             "still comes first: web material is labelled `external: true` in the "
             "payload, carries none of the corpus' licence terms, and the model is told "
             "to cite the url, never to credit a sky culture for it, and to show both "
             "sides where the two disagree.")
    if not has_key:
        st.caption("TAVILY_API_KEY is not set, so the tool cannot be offered.")
    elif internet:
        st.warning("The answer may contain material from outside the 34 cultures. "
                   "Anything from the web is cited by url, not credited to a culture.",
                   icon="🌐")

    st.divider()
    st.caption("Grafana: http://localhost:3000")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])

question = st.chat_input("Who else sees a figure in the Pleiades?")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Asking the corpus..."):
            try:
                answer, run_id, verdicts, seconds = record.ask_and_record(
                    question, lang=lang, internet=internet, images=True)
            except Exception as error:  # noqa: BLE001 - shown, not swallowed
                # A failed run is worth seeing in full: the two failures this catches in
                # practice are a missing API key and a Postgres that is not up yet, and
                # both are fixed by reading the message.
                st.error(f"{type(error).__name__}: {error}")
                st.stop()

        st.write(answer.text)
        show_images(answer)

        cost = f"${answer.cost:.4f}" if answer.cost is not None else "unpriced"
        columns = st.columns(4)
        columns[0].metric("Time", f"{seconds:.1f}s")
        columns[1].metric("In", f"{answer.input_tokens:,}")
        columns[2].metric("Out", f"{answer.output_tokens:,}")
        columns[3].metric("Cost", cost)

        used_the_web = "search_web" in answer.tools_used
        label = ", ".join(answer.tools_used) or "no tools"
        with st.expander(f"Trajectory — {label}" + (" 🌐" if used_the_web else "")):
            if internet and not used_the_web:
                # Worth saying out loud: the run is recorded as `internet = true` either
                # way, and "offered and declined" is the outcome the flag exists to
                # distinguish from "never offered".
                st.caption("The web was available for this question and the model stayed "
                           "in the corpus.")
            for call in answer.calls:
                st.code(f"{call.tool}({call.arguments})", language="python")
            for verdict in verdicts:
                mark = "✅" if verdict["passed"] else "❌"
                st.write(f"{mark} `{verdict['evaluator']}` — {verdict['reason']}")

    st.session_state.history.append({"question": question, "answer": answer.text})
    st.session_state.run_id = run_id

# Outside the `if`, so the buttons survive the rerun a click causes. Keyed by run id so
# two answers in one session cannot collect each other's votes.
run_id = st.session_state.get("run_id")
if run_id is not None:
    st.divider()
    st.caption("Was this answer any good?")
    up, down, _ = st.columns([1, 1, 6])
    if up.button("👍", key=f"up_{run_id}"):
        store.save_verdict(run_id, source="user", evaluator="thumbs", score=1)
        st.success("Recorded. Thank you.")
    if down.button("👎", key=f"down_{run_id}"):
        store.save_verdict(run_id, source="user", evaluator="thumbs", score=-1)
        st.success("Recorded. Thank you — that is the more useful one.")
