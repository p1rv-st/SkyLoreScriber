"""The dense half of retrieval: text to vectors, via ONNX Runtime.

    python -m skylore.query.embed --list
    python -m skylore.query.embed --model e5-small          # build vectors into corpus.db
    python -m skylore.query.embed --model e5-small --langs en

Optional. Nothing else in the package imports this module, and `search_lore` runs
without it, so the whole project stays installable with no dependencies at all
(`pip install -e .` versus `.[embed]`).

**ONNX Runtime rather than torch.** The workload is 1020 vectors built once plus one
short vector per query -- no GPU, no CUDA wheels. The candidate weights are published as
ONNX upstream, so torch is not needed even to convert them.

**What `sentence-transformers` used to do for us, and now we do here.** Pooling: e5
averages the token vectors under the attention mask, bge-m3 takes the CLS position, and
using the wrong one produces vectors that look fine and rank badly. Prefixes: e5 was
trained with `query: ` and `passage: ` and degrades *silently* without them. Both live
in `embedding_models` next to the vectors rather than in code, because a vector built
under one convention and queried under another is indistinguishable from a bad model.

**Long sections become windows.** 54 of 1020 exceed a 512-token context. They are split
into overlapping windows, each embedded separately, and the best-scoring window stands
for the section at query time. The unit that is *returned* is still the whole section --
windows carry vectors and never surface.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import lang
from ..paths import DATABASE
from . import retrieval


@dataclass(frozen=True)
class ModelSpec:
    """Everything about a model that changes the vectors it produces."""
    name: str            # our key, stored in `embeddings.model`
    repo: str
    filename: str
    dim: int
    pooling: str         # mean | cls
    query_prefix: str
    text_prefix: str
    max_tokens: int
    note: str


MODELS: dict[str, ModelSpec] = {
    "e5-small": ModelSpec(
        name="e5-small", repo="intfloat/multilingual-e5-small",
        filename="onnx/model.onnx", dim=384, pooling="mean",
        query_prefix="query: ", text_prefix="passage: ", max_tokens=512,
        note="470 MB. The cheap end -- start here to prove the pipeline.",
    ),
    "e5-base": ModelSpec(
        name="e5-base", repo="intfloat/multilingual-e5-base",
        filename="onnx/model.onnx", dim=768, pooling="mean",
        query_prefix="query: ", text_prefix="passage: ", max_tokens=512,
        note="1.1 GB.",
    ),
    "e5-large": ModelSpec(
        name="e5-large", repo="intfloat/multilingual-e5-large",
        filename="onnx/model.onnx", dim=1024, pooling="mean",
        query_prefix="query: ", text_prefix="passage: ", max_tokens=512,
        note="2.3 GB, weights in a separate .onnx_data file.",
    ),
    "bge-m3": ModelSpec(
        name="bge-m3", repo="BAAI/bge-m3",
        filename="onnx/model.onnx", dim=1024, pooling="cls",
        query_prefix="", text_prefix="", max_tokens=8192,
        note="2.3 GB. CLS pooling, no prefixes, and an 8192-token context -- the only "
             "candidate that embeds every section whole.",
    ),
}


class OnnxEmbedder:
    """Implements `retrieval.Embedder`. The runtime is an implementation detail."""

    def __init__(self, spec: ModelSpec, *, providers: list[str] | None = None):
        import numpy  # noqa: F401  (imported for the side effect of a clear error)
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.spec = spec
        self.name = spec.name
        self.dim = spec.dim

        model_path = hf_hub_download(spec.repo, spec.filename)
        if spec.repo.endswith(("e5-large", "bge-m3")):
            # Weights over 2 GB live beside the graph and must be fetched too; ONNX
            # Runtime loads them by relative path.
            hf_hub_download(spec.repo, spec.filename + "_data")
        self._tokenizer = Tokenizer.from_file(hf_hub_download(spec.repo, "tokenizer.json"))
        self._tokenizer.enable_truncation(spec.max_tokens)

        self._session = onnxruntime.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}

    # ── encoding ──────────────────────────────────────────────────────────────

    def _run(self, texts: list[str]):
        import numpy

        encoded = self._tokenizer.encode_batch(texts)
        width = max(len(e.ids) for e in encoded)
        ids = numpy.zeros((len(encoded), width), dtype=numpy.int64)
        mask = numpy.zeros((len(encoded), width), dtype=numpy.int64)
        for row, item in enumerate(encoded):
            ids[row, : len(item.ids)] = item.ids
            mask[row, : len(item.attention_mask)] = item.attention_mask

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = numpy.zeros_like(ids)
        feed = {k: v for k, v in feed.items() if k in self._inputs}

        hidden = self._session.run(None, feed)[0]
        if self.spec.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            weights = mask[..., None].astype(hidden.dtype)
            pooled = (hidden * weights).sum(axis=1) / weights.sum(axis=1).clip(min=1e-9)
        return pooled

    def encode_query(self, text: str) -> bytes:
        return self.encode_texts([text], prefix=self.spec.query_prefix)[0]

    def encode_texts(self, texts: list[str], prefix: str | None = None) -> list[bytes]:
        body = self.spec.text_prefix if prefix is None else prefix
        pooled = self._run([body + t for t in texts])
        return [retrieval.pack([float(v) for v in row]) for row in pooled]

    def windows(self, text: str) -> list[str]:
        """Split only what does not fit, with overlap so nothing falls between windows."""
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        budget = self.spec.max_tokens - 16          # room for prefix and specials
        if len(ids) <= budget:
            return [text]
        stride = budget // 2                        # 50% overlap
        return [
            self._tokenizer.decode(ids[start:start + budget])
            for start in range(0, len(ids), stride)
            if start == 0 or start + stride < len(ids)
        ]


# ─────────────────────────────────── building ───────────────────────────────────

def embed_text(culture_id: str, region: str | None, heading_path: str, text: str) -> str:
    """The string that actually gets embedded.

    The heading path is prefixed because a section reading "the rainy season begins when
    it sets" is unattributable on its own. This is what replaces the context that chunk
    overlap normally provides.
    """
    where = f"{culture_id} ({region})" if region else culture_id
    return f"Culture: {where} — Section: {heading_path}\n\n{text}"


def build(connection: sqlite3.Connection, embedder: OnnxEmbedder, *,
          langs: tuple[str, ...], batch: int = 16, progress=None) -> dict[str, int]:
    rows = connection.execute(
        f"""SELECT s.id, s.culture_id, c.region, s.heading_path, s.text, s.lang
              FROM sections s JOIN cultures c ON c.id = s.culture_id
             WHERE s.retrievable = 1 AND s.lang IN ({",".join("?" * len(langs))})
             ORDER BY s.id""", langs).fetchall()

    connection.execute("DELETE FROM embeddings WHERE model = ?", (embedder.name,))

    pending: list[tuple[int, int, str]] = []
    stats = {"sections": len(rows), "vectors": 0, "windowed": 0}
    for section_id, culture_id, region, heading_path, text, _ in rows:
        chunks = embedder.windows(embed_text(culture_id, region, heading_path, text))
        stats["windowed"] += len(chunks) > 1
        for index, chunk in enumerate(chunks):
            pending.append((section_id, index, chunk))

    for start in range(0, len(pending), batch):
        group = pending[start:start + batch]
        vectors = embedder.encode_texts([chunk for _, _, chunk in group])
        connection.executemany(
            "INSERT INTO embeddings(section_id, model, window, dim, vector)"
            " VALUES (?,?,?,?,?)",
            [(section_id, embedder.name, window, embedder.dim, vector)
             for (section_id, window, _), vector in zip(group, vectors)],
        )
        stats["vectors"] += len(group)
        if progress:
            progress(stats["vectors"], len(pending))

    spec = embedder.spec
    connection.execute(
        "INSERT OR REPLACE INTO embedding_models"
        "(model, dim, languages, query_prefix, text_prefix, pooling, built_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (spec.name, spec.dim, json.dumps(list(langs)), spec.query_prefix,
         spec.text_prefix, spec.pooling, datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build dense vectors into corpus.db")
    parser.add_argument("--model", choices=sorted(MODELS), help="which model to build")
    parser.add_argument("--langs", nargs="*", default=None,
                        help="languages to embed; default all four")
    parser.add_argument("--list", action="store_true", help="show the candidates")
    args = parser.parse_args(argv)

    if args.list or not args.model:
        for spec in MODELS.values():
            print(f"  {spec.name:10} dim={spec.dim:<5} ctx={spec.max_tokens:<5} "
                  f"{spec.pooling:4}  {spec.note}")
        return 0

    connection = sqlite3.connect(DATABASE)
    langs = tuple(args.langs) if args.langs else lang.available_langs(connection)
    print(f"building {args.model} over {langs}…")

    embedder = OnnxEmbedder(MODELS[args.model])

    def progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total} vectors", end="", flush=True)

    stats = build(connection, embedder, langs=langs, progress=progress)
    print(f"\n  {stats['sections']} sections → {stats['vectors']} vectors "
          f"({stats['windowed']} needed windowing)")
    connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
