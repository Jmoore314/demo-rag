"""
Stage 4: Retrieval logic.

Takes a natural-language question, embeds it the same way we embedded the
document chunks in Stage 2 -- same model, but input_type="query" this time
-- and asks Postgres/pgvector for the chunks whose embeddings are closest
to the question's embedding. This is the step that turns "a pile of
vectors" into "the specific passages relevant to what someone just asked."
"""

import os
import time

import psycopg2
import voyageai
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# Must match the model used to embed the corpus in Stage 2. Mixing
# embedding models would put query and document vectors in different,
# mutually incomparable vector spaces -- distances between them would be
# meaningless, not just less accurate.
MODEL = "voyage-4"

# How many chunks to hand back. Same sizing trade-off logic as chunk size
# in Stage 1, one level up: too few and the right passage might not make
# the cut; too many and Stage 5 has to sift a bloated, partly-irrelevant
# context (cost, and the "lost in the middle" effect we discussed). 3-5 is
# a reasonable starting point for chunks this size (~450 tokens avg).
DEFAULT_TOP_K = 5


# Same throttle-avoidance story as Stage 2: a new Voyage account with no
# payment method is capped at 3 requests/minute. A single query embed is
# tiny either way (well under the 10K TPM cap), but calling retrieve()
# repeatedly in a tight loop -- like testing several questions in a row,
# which is exactly what this script's __main__ does -- can still trip the
# RPM limit. Reusing the same pacing + backoff approach here.
MIN_SECONDS_BETWEEN_REQUESTS = 21
MAX_RETRIES = 5
_last_request_time = 0.0


def _pace_requests():
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    wait = MIN_SECONDS_BETWEEN_REQUESTS - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()


def embed_query(client: voyageai.Client, question: str) -> list[float]:
    delay = 15
    for attempt in range(MAX_RETRIES):
        _pace_requests()
        try:
            result = client.embed([question], model=MODEL, input_type="query")
            return result.embeddings[0]
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"    embed_query failed ({e}); retrying in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60)


def to_pgvector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in embedding) + "]"


# `<=>` is pgvector's cosine *distance* operator: distance = 1 - cosine
# similarity. Ordering ascending by distance is identical to ordering
# descending by similarity -- just the more natural direction to write
# "closest first" in SQL. The HNSW index built in Stage 3 (vector_cosine_ops)
# is what makes Postgres able to answer this without scanning every row
# once the table is large; at 194 rows it wouldn't matter either way.
RETRIEVE_SQL = """
select
    chunk_id,
    source_file,
    page_start,
    page_end,
    content,
    embedding <=> %(query_vector)s::vector as distance
from document_chunks
order by embedding <=> %(query_vector)s::vector asc
limit %(top_k)s
"""


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Returns the top_k chunks most semantically similar to `question`,
    each with a `similarity` score. Because pgvector's <=> is defined as
    exactly (1 - cosine similarity), `1 - distance` recovers the same
    cosine similarity we computed by hand in Stage 2 -- not an
    approximation, the identical quantity."""
    voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    query_vector = to_pgvector_literal(embed_query(voyage, question))

    with psycopg2.connect(os.environ["SUPABASE_DB_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(RETRIEVE_SQL, {"query_vector": query_vector, "top_k": top_k})
            rows = cur.fetchall()

    return [
        {
            "chunk_id": chunk_id,
            "source_file": source_file,
            "page_start": page_start,
            "page_end": page_end,
            "content": content,
            "similarity": 1 - distance,
        }
        for chunk_id, source_file, page_start, page_end, content, distance in rows
    ]


if __name__ == "__main__":
    # Test questions chosen against content we've actually inspected while
    # building this, so we have a real basis to judge "did it retrieve the
    # right thing" -- not just "did it return something."
    test_questions = [
        "What should be entered if a previous cytology result is not known?",
        "What does a lab code represent?",
        "What is required in the PID segment for patient identification?",
    ]

    for q in test_questions:
        print(f"\n{'=' * 80}\nQ: {q}\n{'=' * 80}")
        for i, r in enumerate(retrieve(q, top_k=3), start=1):
            print(f"\n  [{i}] similarity={r['similarity']:.4f}  pages {r['page_start']}-{r['page_end']}  ({r['chunk_id']})")
            snippet = r["content"][:300].replace("\n", " ")
            print(f"      {snippet}...")
