# Reference -- tech stack & environment

---

## Tech stack & packages

| Tool/Package | Role in this project | Why chosen |
|---|---|---|
| `pypdf` | Extracts text (and page numbers) from the source PDF | Pure-Python, no external binary dependency, good enough for text-based PDFs |
| `psycopg2-binary` | Python driver for talking to Postgres | Standard, mature Postgres client for Python |
| `voyageai` | Python client for Voyage AI's embedding API | Chosen embedding provider — Anthropic's recommended embedding partner, pairs well with Claude |
| `anthropic` | Python client for the Claude API | Used in Stage 5 to generate the final grounded answer |
| `python-dotenv` | Loads `.env` file contents into environment variables | Keeps API keys out of source code and out of chat |
| Supabase (Postgres + pgvector extension) | Hosted vector database | User already knows SQL/Postgres; pgvector adds vector similarity search to plain Postgres; hosted because this dev sandbox has no Docker/sudo to run Postgres locally |
| `langchain` / `langchain-community` / `langchain-text-splitters` | Core framework, document loaders, text splitters | Used for the LangChain rebuild -- `PyPDFLoader` and `RecursiveCharacterTextSplitter` replace Stage 1's hand-written extraction/chunking |
| `langchain-postgres` | `PGVector` vector store | Replaces Stage 3's hand-written pgvector schema/upsert SQL; requires psycopg v3 (`psycopg[binary]`), installed alongside psycopg2 which the manual pipeline still uses |
| `langchain-voyageai` | `VoyageAIEmbeddings` wrapper | Replaces Stage 2's direct `voyageai` client calls -- automatically handles the `input_type` document/query distinction, but not rate-limit pacing (see LangChain rebuild section above) |
| `langchain-anthropic` | `ChatAnthropic` chat model wrapper | Replaces Stage 5's direct `anthropic` client call, composed via LCEL |

---

## Environment notes

- Dev sandbox has no `sudo`/Docker access, so a local Postgres server isn't
  possible here — hence the hosted Supabase project for Stage 3.
- Outbound network is allowlisted; some hosts (e.g. Azure blob storage,
  which tiktoken uses) are blocked. PyPI is reachable, so `pip install`
  works normally.

