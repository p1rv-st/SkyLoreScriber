"""Where everything lives. One module, so that moving data is a one-line change.

Before this existed, `corpus.db` was spelled out in nine files -- three library modules
and six test modules -- the corpus directory in two, and `allowlist.json` in two. That is
not a tidiness complaint: it is why the first reorganisation touched a dozen files, and it
is exactly the cost that makes a second one not worth doing. Paths are configuration, and
configuration belongs in one place.

The layout the constants describe:

    data/
      data_to_ingest/stellarium-skycultures/   the submodule, pinned to a commit
      corpus.db                                built by `python -m skylore.ingest`
      allowlist.json  licenses.json            written by `scripts/scan_licenses.py`
      eval/gold.json                           the gold set

`data_to_ingest` carries a directory per upstream source rather than being one: the
corpus is Stellarium's today, and the licence scan records provenance by path
(`licenses.json` names `stellarium-skycultures/*/description.md`), so the source's own
name has to stay in it.

`SCHEMA` is the one path that is *not* data. It ships inside the package because it is
code -- the shape the database is built to -- and it sits in `skylore/ingest/` beside
`build.py`, the only module that applies it, rather than travelling with the database it
produced.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent

DATA = ROOT / "data"
DATA_TO_INGEST = DATA / "data_to_ingest"
CORPUS_DIR = DATA_TO_INGEST / "stellarium-skycultures"

DATABASE = DATA / "corpus.db"
ALLOWLIST = DATA / "allowlist.json"
LICENSES = DATA / "licenses.json"

EVAL = DATA / "eval"
GOLD = EVAL / "gold.json"

SCHEMA = PACKAGE / "ingest" / "schema.sql"

# The path the licence scan records inside `licenses.json`, and the one the AGPL notice
# refers to. Relative to the repository root and written out rather than derived, because
# it appears in generated files that outlive any particular checkout.
CORPUS_NAME = "stellarium-skycultures"
