# RAG from Scratch

A small, working Retrieval-Augmented Generation pipeline, built by hand to
understand every stage end-to-end before reaching for a framework.

Source document: a 140-page HL7 lab-interface technical requirements PDF
(not included in this repo -- see note below). Architecture: Voyage AI for
embeddings, Postgres + pgvector (hosted on Supabase) for vector storage,
Claude for generation.

**The full write-up -- what each stage does, why it's built that way, key
terms, and every real bug hit along the way -- lives in
[`LEARNING_JOURNAL.md`](./LEARNING_JOURNAL.md).** That file is the actual
narrative; this README is just a map.

## Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1. Ingestion & chunking | `src/ingest.py` | Extracts text per page, splits into ~500-token chunks with overlap, forces boundaries at section headers |
| 2. Embeddings | `src/embed.py` | Embeds each chunk with Voyage AI (`input_type="document"`), rate-limit-safe |
| 3. Vector storage | `src/load_vectors.py` | Loads embeddings into a Postgres table with a pgvector HNSW index |
| 4. Retrieval | `src/retrieve.py` | Embeds a query (`input_type="query"`) and does cosine-similarity top-k search |
| 5. Generation | `src/generate.py` | Builds a grounded prompt from retrieved chunks and calls Claude, with citation + refusal rules |

`src/visualize_embeddings.py` is an exploratory, non-pipeline script that
projects the embeddings to 2D (PCA + t-SNE) to sanity-check clustering.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Requires a `.env` file (not committed) with `VOYAGE_API_KEY`,
`ANTHROPIC_API_KEY`, `SUPABASE_DB_URL`, and `SOURCE_PDF_PATH` (the local
path to your own PDF -- kept out of tracked code on purpose, see below).

## What's intentionally not in this repo

- **The source PDF and related spec documents.** They're from a past
  employer project and aren't mine to publish, so the pipeline is shown
  here without its input document. Point `src/ingest.py` at any PDF of
  your own to run it end-to-end.
- **Generated pipeline output** (`data/*.jsonl`) -- chunks and embeddings
  are large and fully reproducible by running the pipeline; only the
  embeddings visualization (`data/embeddings_plot.png`) is kept, as a
  quick look at what the output looks like.

## Status

Manual pipeline: complete, verified against real retrieval and generation
output. A LangChain rebuild of the same five stages -- to compare what a
standard framework provides out of the box versus what still has to be
hand-built -- is planned as a follow-up (see the journal's progress
checklist).
