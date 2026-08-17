"""The RAG loop, and the one tool that leaves the corpus.

Deliberately empty of code. `loop` imports `pydantic_ai` and `web` does not, so a caller
that only wants to know whether web search is enabled should not have to install the
`agent` extra to ask. Import the module you need:

    from skylore.agent import loop     # needs the `agent` extra
    from skylore.agent import web      # needs nothing
"""
