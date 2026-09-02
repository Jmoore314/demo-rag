"""
Stage 3: Loading embedded chunks into Supabase Postgres (pgvector).

The schema (created separately via migration):

    create table document_chunks (
        id           bigserial primary key,
        chunk_id     text not null unique,
        source_file  text not null,
        page_start   integer not null,
        page_end     integer not null,
        token_count  integer not null,
        content      text not null,
        embedding    vector(1024) not null,   -- pgvector's native vector type
        embedding_model text not null,
        created_at   timestamptz not null default now()
    );

Why a native `vector` column instead of just storing the embedding as JSON
or a plain float array? Because pgvector adds similarity *operators*
(<->  euclidean, <#>  negative inner product, <=>  cosine distance) and
index types (HNSW, IVFFlat) that only work against the vector type -- a
JSON column would force a full-table scan with application-side math for
every query, which is exactly what we're using Postgres for pgvector to
avoid.
"""

import json
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")


def _require_env(*names: str) -> None:
    """Fail with a plain, actionable message instead of a bare KeyError --
    this repo is meant to be read and adapted, not cloned-and-run without
    your own document and API keys (see README.md's Setup section)."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"Missing required .env value(s): {', '.join(missing)}")
        print("See README.md's Setup section -- this pipeline needs your own API keys.")
        sys.exit(1)


def to_pgvector_literal(embedding: list[float]) -> str:
    """pgvector accepts vector input as a bracketed string like
    '[0.01,-0.02,...]', cast to the vector type. We build that string
    ourselves rather than pulling in the separate `pgvector` Python package
    -- one less dependency for something this simple."""
    return "[" + ",".join(repr(x) for x in embedding) + "]"


def load_chunks_embedded(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


UPSERT_SQL = """
insert into document_chunks
    (chunk_id, source_file, page_start, page_end, token_count, content, embedding, embedding_model)
values %s
on conflict (chunk_id) do update set
    source_file     = excluded.source_file,
    page_start      = excluded.page_start,
    page_end        = excluded.page_end,
    token_count     = excluded.token_count,
    content         = excluded.content,
    embedding       = excluded.embedding,
    embedding_model = excluded.embedding_model
"""

# execute_values needs an explicit ::vector cast in the template, since the
# driver would otherwise send the bracketed string as plain text and
# Postgres has no default text->vector cast for an untyped literal in this
# position.
VALUE_TEMPLATE = "(%(chunk_id)s, %(source_file)s, %(page_start)s, %(page_end)s, %(token_count)s, %(content)s, %(embedding)s::vector, %(embedding_model)s)"


def main():
    _require_env("SUPABASE_DB_URL")
    if not Path("data/chunks_embedded.jsonl").exists():
        print("data/chunks_embedded.jsonl not found -- run embed.py first.")
        sys.exit(1)
    db_url = os.environ["SUPABASE_DB_URL"]
    records = load_chunks_embedded("data/chunks_embedded.jsonl")
    print(f"Loaded {len(records)} chunk records from disk.")

    rows = [
        {
            "chunk_id": r["chunk_id"],
            "source_file": r["source_file"],
            "page_start": r["page_start"],
            "page_end": r["page_end"],
            "token_count": r["token_count"],
            "content": r["text"],
            "embedding": to_pgvector_literal(r["embedding"]),
            "embedding_model": r["embedding_model"],
        }
        for r in records
    ]

    print("Connecting to Supabase Postgres...")
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            # TRUNCATE first: the upsert below protects against re-running
            # the SAME chunking twice (safe, idempotent). It does NOT
            # protect against a re-CHUNKING, where chunk boundaries shifted
            # and old chunk_ids no longer correspond to the same content --
            # without clearing first, we'd end up with old and new chunks
            # coexisting in the table, most with stale/wrong text.
            cur.execute("truncate table document_chunks")
            psycopg2.extras.execute_values(cur, UPSERT_SQL, rows, template=VALUE_TEMPLATE, page_size=50)
        conn.commit()

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*), min(page_start), max(page_end) from document_chunks")
            count, min_page, max_page = cur.fetchone()

    print(f"Done. document_chunks now has {count} rows, spanning pages {min_page}-{max_page}.")


if __name__ == "__main__":
    main()
