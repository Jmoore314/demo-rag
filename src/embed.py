"""
Stage 2: Generating embeddings for chunks with Voyage AI.

An embedding is a fixed-length vector of floats (for our model, 1024 numbers)
produced by a neural network trained so that texts with similar *meaning*
end up as vectors that are numerically close together in that 1024-dimensional
space -- regardless of whether they share any of the same words. That's the
whole trick that makes RAG retrieval possible: instead of keyword matching,
we compare vectors, and "close vectors" approximates "related meaning."
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

import voyageai

# voyage-4 is Voyage's current general-purpose embedding model (their law/
# finance/code models are domain-tuned variants we don't need here). 1024
# output dimensions by default, 32K token context per input, and the first
# 200M tokens are free -- our ~87K-token corpus costs nothing.
MODEL = "voyage-4"

# Voyage's API allows up to 1000 texts per request in principle, but a
# fresh account with no payment method on file is rate-limited to 3
# requests/minute and 10K tokens/minute (the free 200M-token allowance
# still applies regardless -- this only throttles speed, not cost). At
# ~450 tokens/chunk average, batches of 15 stay safely under the 10K TPM
# ceiling even for a batch that happens to be all larger chunks.
BATCH_SIZE = 15
MAX_RETRIES = 5

# 60s / 3 requests = 20s minimum spacing; we pad it slightly.
MIN_SECONDS_BETWEEN_REQUESTS = 21


def _require_env(*names: str) -> None:
    """Fail with a plain, actionable message instead of a bare KeyError --
    this repo is meant to be read and adapted, not cloned-and-run without
    your own document and API keys (see README.md's Setup section)."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"Missing required .env value(s): {', '.join(missing)}")
        print("See README.md's Setup section -- this pipeline needs your own API keys.")
        sys.exit(1)


def load_chunks(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _batched(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


_last_request_time: float = 0.0


def _pace_requests():
    """Proactively space requests out to respect the 3 RPM ceiling, rather
    than firing as fast as possible and relying on retries to recover from
    429s. Politer to the API, and faster overall since we're not wasting
    time on rejected calls."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    wait = MIN_SECONDS_BETWEEN_REQUESTS - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()


def _embed_batch_with_retry(client: voyageai.Client, texts: list[str], input_type: str) -> list[list[float]]:
    delay = 15  # rate-limit errors need a real wait, not a quick retry
    for attempt in range(MAX_RETRIES):
        _pace_requests()
        try:
            result = client.embed(texts, model=MODEL, input_type=input_type)
            return result.embeddings
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"    batch failed ({e}); retrying in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60)


def embed_texts(client: voyageai.Client, texts: list[str], input_type: str) -> list[list[float]]:
    """input_type is not cosmetic. Voyage silently prepends a different
    instruction string internally depending on whether you say
    input_type="document" (the corpus side, used here) or
    input_type="query" (the user's question side, used in Stage 4). This
    "asymmetric" embedding is deliberate -- a question and its answer often
    don't look alike as raw text ("How do I handle an unknown cytology
    result?" vs "ZAB-13 ... If the answer is not known this field should be
    blank"), so the model is trained to embed queries and documents into
    matching regions of the vector space *despite* that surface mismatch,
    but only if you tell it which side is which. Get this backwards (or
    skip it) and nothing crashes -- retrieval just quietly gets worse.
    """
    all_embeddings: list[list[float]] = []
    n_batches = -(-len(texts) // BATCH_SIZE)  # ceiling division
    for i, batch in enumerate(_batched(texts, BATCH_SIZE), start=1):
        print(f"  batch {i}/{n_batches} ({len(batch)} texts)...")
        all_embeddings.extend(_embed_batch_with_retry(client, batch, input_type))
    return all_embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    _require_env("VOYAGE_API_KEY")
    if not Path("data/chunks.jsonl").exists():
        print("data/chunks.jsonl not found -- run ingest.py first.")
        sys.exit(1)
    chunks = load_chunks("data/chunks.jsonl")
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    print(f"Embedding {len(chunks)} chunks with {MODEL} (input_type=document)...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(client, texts, input_type="document")
    assert len(embeddings) == len(chunks)

    out_path = Path("data/chunks_embedded.jsonl")
    with out_path.open("w") as f:
        for chunk, vec in zip(chunks, embeddings):
            record = dict(chunk)
            record["embedding"] = vec
            record["embedding_model"] = MODEL
            f.write(json.dumps(record) + "\n")

    dim = len(embeddings[0])
    print(f"\nDone. {len(embeddings)} embeddings, dimension={dim}. Wrote {out_path}")

    # --- Sanity check: does "close in vector space" actually track meaning? ---
    print("\n=== Sanity check: cosine similarity between chunk pairs ===")

    def show(i, j, label):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{label}: chunk-{i:04d} vs chunk-{j:04d}  cosine similarity = {sim:.4f}")

    # Adjacent chunks (216-chunk corpus, both from the dense HL7 field-code
    # section) -- these share overlap text AND topic, so similarity should
    # be high.
    show(80, 81, "Adjacent, same topic (HL7 field codes)")

    # Far-apart chunks on unrelated topics (front matter/TOC vs a deep field
    # spec section) -- similarity should be noticeably lower.
    show(2, 80, "Unrelated topics (front matter vs field codes)")
