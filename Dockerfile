# The Streamlit chat. Postgres and Grafana are stock images and need no Dockerfile.
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Dependencies before source, so editing a module does not reinstall the world. `--locked`
# because `uv.lock` is committed: the image resolves to what the tests ran against, not to
# whatever was newest the morning it was built.
#
# `--extra agent --extra monitor` and deliberately not `--extra embed`: the dense half of
# `search_lore` is 120 MB of ONNX Runtime plus a 2.3 GB model, and `search_lore` falls back
# to BM25 alone without it. An image four times its useful size to improve one tool by one
# gold question is a bad trade; anyone who wants it can add the extra here and rebuild.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --extra agent --extra monitor

COPY skylore/ ./skylore/

# `data/` is mounted rather than copied. corpus.db is 7 MB and rebuildable, but the
# submodule it is built from is 168 MB, and a container that carried the sources it no
# longer needs would be paying for the ingest step at every pull.
EXPOSE 8501
CMD ["streamlit", "run", "skylore/monitor/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
