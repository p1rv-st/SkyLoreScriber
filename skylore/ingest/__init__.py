"""Building the database from the corpus. `python -m skylore.ingest`

`build` is the pass itself; `corpus` and `po` are the readers it runs on, and they are
useful on their own -- `corpus.read_index` and `corpus.cited_refs` are called from the
licence scan and the query layer respectively. Nothing is re-exported here: a caller
names the module it wants, so an import says which of the three it depends on.
"""
