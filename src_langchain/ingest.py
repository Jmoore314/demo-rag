"""
LangChain rebuild -- Stage 1: document ingestion & chunking.

Direct comparison point for src/ingest.py. Two LangChain pieces replace
everything in the manual version except the header-flush fix:

  - PyPDFLoader                       replaces extract_pages() / clean_text()
  - RecursiveCharacterTextSplitter    replaces chunk_pages()'s hand-written
                                       paragraph -> sentence -> hard-cut
                                       fallback (_split_oversized())

Known, deliberate differences from src/ingest.py (full writeup in
LEARNING_JOURNAL.md):

  1. Page boundaries. The manual pipeline concatenates the whole document
     into one text stream before chunking, so a chunk CAN span a page
     break if that's where a paragraph naturally falls. PyPDFLoader loads
     one Document per page, and split_documents() splits each page's
     Document independently -- so here, no chunk can ever span two pages.
     If a sentence is cut in half by a page break in the source PDF, this
     version cuts the chunk there too. This is a real, testable
     difference, not an oversight -- worth checking whether it costs
     anything once retrieval results come back.
  2. No header-flush fix. RecursiveCharacterTextSplitter has no concept
     of "this line looks like a section heading, force a boundary here"
     -- that heuristic (_HEADER_RE / _split_on_headers) was hand-written
     for src/ingest.py specifically, after the fact, to fix a dilution
     problem found empirically. LangChain chunks purely on size + generic
     separators (paragraph, line, sentence, word, character) here, so
     this version should reproduce the SAME embedding-dilution symptom
     the header-flush fix solved (e.g. the PID-segment query). That's a
     concrete, falsifiable prediction for Stage 4 of this rebuild to
     confirm or contradict.

Chunk size, overlap, and the token-counting heuristic are kept IDENTICAL
to the manual pipeline on purpose, so the comparison isn't confounded by
two different chunking parameters on top of everything else.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=".env")

# Same env-var indirection as src/ingest.py -- the real filename lives only
# in the local, gitignored .env as SOURCE_PDF_PATH.
PDF_PATH = os.environ.get("SOURCE_PDF_PATH", "EDI Specifications/source-spec.pdf")
OUT_PATH = "data/langchain_chunks.jsonl"


# Same heuristic as src/ingest.py's count_tokens(): tiktoken's real
# tokenizer needs a network download this sandbox can't make, and it's
# not even Voyage's/Claude's actual tokenizer anyway -- ~4 chars/token is
# the same honest approximation used throughout this project. Passed to
# the splitter as `length_function` so CHUNK_SIZE/CHUNK_OVERLAP below are
# measured in the same units (approx. tokens) as the manual pipeline,
# not raw characters.
def count_tokens(text: str) -> int:
    return len(text) // 4


CHUNK_SIZE = 500       # target tokens per chunk -- same as src/ingest.py
CHUNK_OVERLAP = 75     # ~15% overlap -- same as src/ingest.py


def ingest_pdf(pdf_path: str = PDF_PATH):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # one Document per page; metadata["page"] is 0-indexed
    print(f"Loaded {len(pages)} pages via PyPDFLoader")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=count_tokens,
        # Default separator list already tries paragraph -> line ->
        # sentence -> word -> character, in that order -- this is the
        # built-in equivalent of src/ingest.py's _split_oversized().
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def main():
    chunks = ingest_pdf()

    token_counts = [count_tokens(c.page_content) for c in chunks]
    print(
        f"Token range: {min(token_counts)}-{max(token_counts)} "
        f"(avg {sum(token_counts) / len(token_counts):.0f})"
    )

    Path("data").mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for i, chunk in enumerate(chunks):
            record = {
                "chunk_id": f"lc-chunk-{i:04d}",
                "text": chunk.page_content,
                # store 1-indexed to match the manual pipeline's page_start
                "page": chunk.metadata.get("page", -1) + 1,
                "token_count": count_tokens(chunk.page_content),
            }
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {OUT_PATH}")

    print("\n--- Sample chunk #5 ---")
    print(chunks[5].page_content[:400])
    print(
        f"...\n[page {chunks[5].metadata.get('page', -1) + 1}, "
        f"{count_tokens(chunks[5].page_content)} tokens]"
    )


if __name__ == "__main__":
    main()
