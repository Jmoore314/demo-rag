"""
LangChain rebuild -- Stage 4: retrieval logic.

Direct comparison point for src/retrieve.py. `vector_store.
similarity_search_with_score_by_vector()` replaces retrieve()'s
hand-written SQL for the actual pgvector query.

**Real bug hit running this, not a hypothetical:** the obvious, idiomatic
call here is `vector_store.similarity_search_with_score(question, k=k)`,
which embeds the query internally via `self.embeddings.embed_query()`.
First real run threw `voyageai.error.RateLimitError` on the very first
question. Root cause, same category of gap load_vectors.py already found
in VoyageAIEmbeddings: `embed_query()` calls the Voyage client directly
with no pacing and no meaningful retry -- it has nothing to protect a
free-tier 3 RPM account, especially right after load_vectors.py's own
bulk-embedding run, which leaves the account's rate-limit window still
hot. This is the LangChain-rebuild version of a problem the MANUAL
pipeline already hit and fixed in the exact same place: src/retrieve.py
needed the same pacing logic as src/embed.py for precisely this reason
(see LEARNING_JOURNAL.md, Stage 5 section, item 5). Same lesson,
rediscovered independently in the rebuild -- strong evidence it's a real
property of the account/API, not a fluke of one pipeline's code.

Fix: embed the query ourselves with the same paced/retried call used for
the bulk corpus in load_vectors.py, then hand the vector directly to
`similarity_search_with_score_by_vector()` instead of letting PGVector
call embed_query() unpaced internally.

**Also worth restating (verified via source, see load_vectors.py):**
the score returned is pgvector COSINE DISTANCE, not similarity, despite
the method name -- `1 - score` below converts it to the same
`similarity` convention src/retrieve.py uses, so the two pipelines'
numbers are directly comparable.
"""

import os
import time

from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv(dotenv_path=".env")

import voyageai

MODEL = "voyage-4"
COLLECTION_NAME = "demo_rag_langchain"
DEFAULT_TOP_K = 5

# Same constants as src/embed.py and src_langchain/load_vectors.py.
MAX_RETRIES = 5
MIN_SECONDS_BETWEEN_REQUESTS = 21

_last_request_time: float = 0.0


def _pace_requests():
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    wait = MIN_SECONDS_BETWEEN_REQUESTS - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()


def _embed_query_with_retry(client: voyageai.Client, question: str) -> list[float]:
    delay = 15
    for attempt in range(MAX_RETRIES):
        _pace_requests()
        try:
            result = client.embed([question], model=MODEL, input_type="query")
            return result.embeddings[0]
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"    query embed failed ({e}); retrying in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60)


def to_psycopg3_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_vector_store() -> PGVector:
    # VoyageAIEmbeddings is still constructed and passed in -- PGVector
    # requires an Embeddings object at construction time even though we
    # bypass its embed_query() call below in favor of our own paced one.
    voyage_embeddings = VoyageAIEmbeddings(
        voyage_api_key=os.environ["VOYAGE_API_KEY"],
        model=MODEL,
    )
    connection = to_psycopg3_url(os.environ["SUPABASE_DB_URL"])
    return PGVector(
        embeddings=voyage_embeddings,
        collection_name=COLLECTION_NAME,
        connection=connection,
        use_jsonb=True,
    )


def retrieve(vector_store: PGVector, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    query_vector = _embed_query_with_retry(client, question)

    results = vector_store.similarity_search_with_score_by_vector(query_vector, k=top_k)
    return [
        {
            "chunk_id": doc.id,
            "page": doc.metadata.get("page"),
            "similarity": 1 - distance,  # see module docstring -- distance, not similarity
            "content": doc.page_content,
        }
        for doc, distance in results
    ]


if __name__ == "__main__":
    # Same 3 test questions as src/retrieve.py, so the results are a real
    # apples-to-apples comparison, not just "different questions, hard to
    # tell if anything actually changed."
    test_questions = [
        "What should be entered if a previous cytology result is not known?",
        "What does a lab code represent?",
        "What is required in the PID segment for patient identification?",
    ]

    store = get_vector_store()

    for q in test_questions:
        print(f"\n{'=' * 80}\nQ: {q}\n{'=' * 80}")
        for i, r in enumerate(retrieve(store, q, top_k=3), start=1):
            print(f"\n  [{i}] similarity={r['similarity']:.4f}  page {r['page']}  ({r['chunk_id']})")
            snippet = r["content"][:300].replace("\n", " ")
            print(f"      {snippet}...")
