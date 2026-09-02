# RAG from Scratch

> **This repo is a portfolio artifact, not a clone-and-run tool.** The
> source PDF and the Voyage AI / Anthropic / Supabase credentials it needs
> are intentionally excluded (see "What's intentionally not in this repo"
> below) -- running the scripts as-is will stop with a clear error message
> telling you what's missing, not produce results. Point `SOURCE_PDF_PATH`
> at your own document and supply your own API keys in a local `.env` to
> actually execute it end-to-end.

A small, working Retrieval-Augmented Generation pipeline, built by hand to
understand every stage end-to-end before reaching for a framework.

Source document: a 140-page HL7 lab-interface technical requirements PDF
(not included in this repo -- see note below). Architecture: Voyage AI for
embeddings, Postgres + pgvector (hosted on Supabase) for vector storage,
Claude for generation.

**The full write-up -- what each stage does, why it's built that way, key
terms, and every real bug hit along the way -- lives in the
[`journal/`](./journal/) directory, starting at
[`journal/00-overview.md`](./journal/00-overview.md).** Those files are the
actual narrative; this README is just a map.

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

## LangChain rebuild

The same five stages, rebuilt in `src_langchain/` using LangChain instead
of hand-written code, to compare what a standard framework provides out of
the box versus what still has to be built by hand regardless. Same `.env`,
same source PDF, a separate Postgres collection so it never touches the
manual pipeline's data. Run in this order:

| Stage(s) | Script | Replaces |
|---|---|---|
| 1. Ingestion & chunking | `src_langchain/ingest.py` | `PyPDFLoader` + `RecursiveCharacterTextSplitter` replace `extract_pages()`/`clean_text()`/`chunk_pages()` |
| 2+3. Embeddings & vector storage | `src_langchain/load_vectors.py` | `VoyageAIEmbeddings` + `PGVector` replace `embed.py` + `load_vectors.py` (these two collapse into one file/step) |
| 4. Retrieval | `src_langchain/retrieve.py` | `PGVector.similarity_search_with_score_by_vector()` replaces the hand-written SQL query |
| 5. Generation | `src_langchain/generate.py` | `ChatPromptTemplate` + `ChatAnthropic`, composed via LCEL, replace `build_prompt()` + the raw API call |

Full comparison -- what LangChain got right, two real gaps found by
reading its installed source, and a chunking-behavior prediction a live
run contradicted -- is in
[`journal/02-langchain-rebuild.md`](./journal/02-langchain-rebuild.md).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `VOYAGE_API_KEY`,
`ANTHROPIC_API_KEY`, `SUPABASE_DB_URL`, and `SOURCE_PDF_PATH` (the local
path to your own PDF -- kept out of tracked code on purpose, see below).

Before running the manual pipeline (`src/`), run [`schema.sql`](./schema.sql)
once against your Postgres database to create the `document_chunks` table
and enable pgvector. The LangChain rebuild (`src_langchain/`) doesn't need
this -- it manages its own table automatically.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

67 tests, no API key or `.env` required, no network calls, no cost --
everything deterministic (chunking, prompt formatting, URL/vector-literal
conversion) is tested directly, and everything that normally calls an API
or a database (retry/pacing logic, `retrieve()`, `generate_answer()`) is
tested against a mocked client instead of a real one. What's deliberately
*not* here -- retrieval/generation quality against the real document --
is a different kind of check than a unit test can give an honest answer
to; see [`journal/05-testing.md`](./journal/05-testing.md) for why, and
what that would look like instead.

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

Manual pipeline and the LangChain rebuild of the same five stages: both
complete, both verified against real retrieval and generation output (see
[`journal/02-langchain-rebuild.md`](./journal/02-langchain-rebuild.md) for
the comparison). A `pytest` suite covers the deterministic and mockable
parts of both (see Testing, above).
