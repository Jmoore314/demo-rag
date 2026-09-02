"""
LangChain rebuild -- Stages 2+3: embeddings + vector storage.

Direct comparison point for src/embed.py + src/load_vectors.py. In
LangChain these two stages naturally collapse into one: a VectorStore is
constructed with an Embeddings object, and its add_documents()/from_
documents() methods embed and store in a single call. That collapse
itself is worth noting -- it's a real design difference from the manual
pipeline's two-separate-files-two-separate-concerns split.

**Why this file does NOT just call PGVector.from_documents(chunks,
embeddings) the "normal" idiomatic way -- a real finding, not a style
choice:**

Read the installed langchain-voyageai source before writing this (see
journal/02-langchain-rebuild.md for the full walkthrough). VoyageAIEmbeddings.
embed_documents() batches texts by calling `self._client.tokenize(...)`
ONCE PER TEXT before it ever calls the actual embed endpoint -- purely to
measure how many texts fit under the model's token-per-batch ceiling. For
216 chunks, that's 216 tokenize() API calls, on top of the ~15 embed()
calls, with zero built-in pacing or retry logic between any of them (the
class just calls the client directly). On this account's 3 RPM/10K TPM
free-tier throttle, that's worse than the manual pipeline's embed.py,
which makes zero tokenize calls at all (it uses the same chars/4
heuristic as ingest.py, purely local, no network round-trip).

So: the bulk corpus is embedded with the SAME hand-written pacing/retry
logic as src/embed.py (copied essentially verbatim below), using the raw
voyageai client directly -- and the resulting vectors are handed to
PGVector.add_embeddings(), which stores pre-computed vectors WITHOUT
calling the embeddings object again. VoyageAIEmbeddings is still
constructed and passed to PGVector, because the vector store needs an
Embeddings object internally for embed_query() at retrieval time (Stage
4) -- one call per question, where the per-text tokenize overhead is
completely negligible. That's the honest place to let the framework do
the work for you: low-volume, latency-insensitive, not rate-limit-prone.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv(dotenv_path=".env")

import voyageai


def _require_env(*names: str) -> None:
    """Fail with a plain, actionable message instead of a bare KeyError --
    this repo is meant to be read and adapted, not cloned-and-run without
    your own document and API keys (see README.md's Setup section)."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"Missing required .env value(s): {', '.join(missing)}")
        print("See README.md's Setup section -- this pipeline needs your own API keys.")
        sys.exit(1)


MODEL = "voyage-4"
COLLECTION_NAME = "demo_rag_langchain"  # separate from the manual pipeline's document_chunks table

# --- Same pacing/retry constants and logic as src/embed.py ---
BATCH_SIZE = 15
MAX_RETRIES = 5
MIN_SECONDS_BETWEEN_REQUESTS = 21  # 60s / 3 RPM, padded slightly

_last_request_time: float = 0.0


def _pace_requests():
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    wait = MIN_SECONDS_BETWEEN_REQUESTS - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()


def _batched(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _embed_batch_with_retry(client: voyageai.Client, texts: list[str]) -> list[list[float]]:
    delay = 15
    for attempt in range(MAX_RETRIES):
        _pace_requests()
        try:
            result = client.embed(texts, model=MODEL, input_type="document")
            return result.embeddings
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"    batch failed ({e}); retrying in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60)


def embed_corpus(texts: list[str]) -> list[list[float]]:
    """Bulk-embed via the raw client + our own pacing -- see module
    docstring for why this bypasses VoyageAIEmbeddings.embed_documents()."""
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    all_embeddings: list[list[float]] = []
    n_batches = -(-len(texts) // BATCH_SIZE)
    for i, batch in enumerate(_batched(texts, BATCH_SIZE), start=1):
        print(f"  batch {i}/{n_batches} ({len(batch)} texts)...")
        all_embeddings.extend(_embed_batch_with_retry(client, batch))
    return all_embeddings


def to_psycopg3_url(url: str) -> str:
    """langchain-postgres's PGVector requires a psycopg v3 connection
    string (postgresql+psycopg://...). SUPABASE_DB_URL in .env is a plain
    postgresql:// URL -- the format psycopg2 (used by the manual pipeline)
    expects. Same database, different driver prefix."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def load_chunks(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    _require_env("VOYAGE_API_KEY", "SUPABASE_DB_URL")
    if not Path("data/langchain_chunks.jsonl").exists():
        print("data/langchain_chunks.jsonl not found -- run src_langchain/ingest.py first.")
        sys.exit(1)

    chunks = load_chunks("data/langchain_chunks.jsonl")
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"page": c["page"], "token_count": c["token_count"]} for c in chunks]

    print(f"Embedding {len(chunks)} chunks with {MODEL} (input_type=document)...")
    embeddings = embed_corpus(texts)
    assert len(embeddings) == len(chunks)

    voyage_embeddings = VoyageAIEmbeddings(
        voyage_api_key=os.environ["VOYAGE_API_KEY"],
        model=MODEL,
    )

    connection = to_psycopg3_url(os.environ["SUPABASE_DB_URL"])
    vector_store = PGVector(
        embeddings=voyage_embeddings,
        collection_name=COLLECTION_NAME,
        connection=connection,
        use_jsonb=True,
    )

    print(f"Storing {len(chunks)} pre-computed embeddings into collection '{COLLECTION_NAME}'...")
    # add_embeddings (not add_documents/from_documents) -- stores the
    # vectors we already computed above instead of re-embedding via
    # VoyageAIEmbeddings.embed_documents(). Passing the same `ids` again
    # on a re-run replaces those rows rather than duplicating them, same
    # idempotency property as the manual pipeline's upsert.
    vector_store.add_embeddings(texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    print(f"Done. Collection '{COLLECTION_NAME}' now has {len(chunks)} rows.")


if __name__ == "__main__":
    main()
