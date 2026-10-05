-- Schema for the `document_chunks` table used by src/load_vectors.py and
-- src/retrieve.py (the manual pipeline). Run once against a Postgres
-- database with the pgvector extension available (Supabase has this
-- built in; enable it per-project below). Not run automatically by any
-- script in this repo -- see journal/01-build-log.md, Stage 3, for why
-- schema/extension/index provisioning was done as one-time infrastructure
-- setup rather than folded into the pipeline code itself.
--
-- The LangChain rebuild (src_langchain/) does not use this table: it
-- writes into a separate collection ("demo_rag_langchain") that
-- langchain-postgres's PGVector class creates and manages on its own.

create extension if not exists vector;

create table if not exists document_chunks (
    id              bigserial primary key,
    chunk_id        text not null unique,
    source_file     text not null,
    page_start      integer not null,
    page_end        integer not null,
    token_count     integer not null,
    content         text not null,
    embedding       vector(1024) not null,  -- matches Voyage AI's voyage-4 output dimension
    embedding_model text not null,
    created_at      timestamptz not null default now()
);

-- Added after the data was loaded and verified (see journal) -- HNSW has
-- no requirement that representative data already exist before building,
-- unlike IVFFlat, which is one reason to prefer it here. The operator
-- class (vector_cosine_ops) must match the distance operator used at
-- query time (<=>, cosine distance) or Postgres silently won't use the
-- index at all.
create index if not exists document_chunks_embedding_hnsw_idx
    on document_chunks
    using hnsw (embedding vector_cosine_ops);

-- Lock the table down at the API layer. With RLS enabled and no policies,
-- the Supabase anon/authenticated keys (PostgREST) cannot read or write
-- any rows. The pipeline connects directly to Postgres via SUPABASE_DB_URL
-- (the table owner role), which bypasses RLS, so it is unaffected.
alter table document_chunks enable row level security;

-- Note: the LangChain rebuild's tables (langchain_pg_collection and
-- langchain_pg_embedding) are created at runtime by PGVector, so they
-- can't be locked down here. After running src_langchain/load_vectors.py
-- for the first time, run:
--   alter table langchain_pg_collection enable row level security;
--   alter table langchain_pg_embedding enable row level security;
