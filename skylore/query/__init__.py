"""Reading the database: names, articles, ranking, and the cross-culture join.

Every module here takes an open connection and returns dataclasses -- no module opens a
database of its own, because a tool call answers one question against one connection and
`skylore.tools` is what owns it.

`embed` is the exception in kind rather than in shape: it *writes* the vectors that
`retrieval` reads. It sits here because the class it builds, `OnnxEmbedder`, is handed
back into `retrieval` at query time, and because it is the one module in the package that
needs the `embed` extra installed. `python -m skylore.query.embed --model bge-m3`
"""
