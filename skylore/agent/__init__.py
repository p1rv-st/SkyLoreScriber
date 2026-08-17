"""The RAG loop, the tool that leaves the corpus, and what a finished run says.

Deliberately empty of code. `loop` imports `pydantic_ai` and the other three do not, so a
caller that only wants to know whether web search is enabled, or to grade a trajectory it
already has, should not have to install the `agent` extra to ask. Import the module you
need:

    from skylore.agent import loop     # needs the `agent` extra
    from skylore.agent import web      # needs nothing
    from skylore.agent import checks   # needs nothing
"""
