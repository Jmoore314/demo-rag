# RAG from Scratch — Learning Journal

A running record of what I built, why I built it that way, and the vocabulary
that goes with it. Source document: a 140-page HL7 lab-interface technical
spec PDF (excluded from this repo -- not mine to publish; see README).
Goal: build a working RAG pipeline by hand first, then rebuild it with
LangChain to see what the framework buys you.

**Architecture chosen:** Voyage AI for embeddings → Supabase Postgres
(pgvector) for storage/retrieval → Claude (Anthropic API) for generation.

---

## Progress

- [x] Stage 1 — Document ingestion & chunking
- [x] Stage 2 — Embeddings (Voyage AI)
- [x] Stage 3 — Vector storage (Supabase + pgvector)
- [x] Stage 4 — Retrieval logic
- [x] Stage 5 — Prompt construction & Claude generation
- [x] Verification pass (folded into Stage 5 testing -- see below)
- [x] LangChain rebuild

**Rough time estimate for the full project (manual build + LangChain
rebuild):** ~5–9 hours of hands-on, explained work, most realistically spread
across a few sessions rather than one sitting. Stage 1 (including debugging
the chunk-size and tiktoken issues below) took roughly 45 minutes. Supabase
setup in Stage 3 will likely be the next-longest stage because of project
provisioning time.


---

## How this journal is organized

This journal grew past the point of being readable in one sitting, so it's
split into a few files, each with a clear job:

- **[`01-build-log.md`](./01-build-log.md)** -- the manual pipeline
  (`src/`), told stage by stage in build order (Stage 1 through Stage 5),
  each stage followed immediately by the vocabulary it introduced. This is
  the core "how I built it, and what went wrong along the way" narrative.
- **[`02-langchain-rebuild.md`](./02-langchain-rebuild.md)** -- the
  LangChain rebuild (`src_langchain/`), compared directly against the
  manual pipeline: what the framework replaced, what it got right, and two
  real gaps found by reading its installed source.
- **[`03-reference.md`](./03-reference.md)** -- quick-lookup tables: the
  tech stack and why each piece was chosen, plus sandbox/environment notes.
- **[`04-system-design-notes.md`](./04-system-design-notes.md)** --
  forward-looking notes on how this project would need to change to scale
  past "one document, one pipeline" (evaluation metrics, chunking
  alternatives, multi-tenant architecture). Explicitly not things built or
  debugged here -- kept separate from the rest of the journal so that
  distinction stays clear.
