"""
Stage 1: Document ingestion & chunking.

Goal: turn a PDF into a list of small, overlapping text chunks that we can
later embed and search over. Everything downstream (embedding quality,
retrieval accuracy, answer grounding) is bottlenecked by decisions made here,
so this stage gets its own file and its own scrutiny.
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv(dotenv_path=".env")

# We don't have the real Voyage or Claude tokenizer available offline, and
# pulling one down (e.g. tiktoken's cl100k_base, which isn't even the exact
# tokenizer either model uses) requires a network call this environment's
# egress allowlist blocks. So we use a well-known rule of thumb instead:
# English text averages ~4 characters per token. It's approximate, but for
# the one thing we need it for -- "is this chunk roughly the right size" --
# approximate is genuinely fine. Precision here wouldn't change any
# downstream decision.
_CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _take_last_tokens(text: str, n_tokens: int) -> str:
    """Approximate 'last n tokens' by characters, then snap to a word
    boundary so we don't start the overlap mid-word."""
    n_chars = n_tokens * _CHARS_PER_TOKEN
    tail = text[-n_chars:]
    # Drop a possibly-truncated leading word.
    first_space = tail.find(" ")
    if first_space != -1:
        tail = tail[first_space + 1 :]
    return tail


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    token_count: int
    text: str


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, raw_text), ...], 1-indexed page numbers.

    We keep page numbers attached to every page's text *before* chunking so
    that later, every chunk can carry "this came from pages 12-13" as
    metadata. That's what lets a generated answer cite a real page number
    instead of just asserting things.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def clean_text(text: str) -> str:
    """Light cleanup only. We deliberately do NOT try to be clever here.

    PDF text extraction is messy: hyphenated line-wraps, stray form-feed
    characters, runs of whitespace from multi-column layouts. We fix the
    cheap, safe stuff and leave the rest -- aggressive "cleanup" regexes are
    a common source of silently mangled technical content (think HL7 segment
    names like "PID-3" getting mashed together).
    """
    text = text.replace("\x0c", " ")  # form feed
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate wrapped words
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Detects section-header-like lines: a leading number (optionally dotted,
# e.g. "5.3.2") followed by a heading in the same line, e.g.
# "3 SUPPLEMENTAL NOTES" or "6.1.4 Example Segment - Sample ID Fields" --
# both real patterns in this document. Deliberately requires heading text
# on the SAME line as the number, which is what keeps this from
# false-matching bare subfield numbers like a lone "5.2" that precedes its
# label on the next line -- those have nothing to match after the digits.
_HEADER_RE = re.compile(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z0-9 ,\-/&]{2,80}$")


def _is_header_line(line: str) -> bool:
    return bool(_HEADER_RE.match(line.strip()))


def _split_on_headers(text: str) -> list[str]:
    """Break a paragraph block into pieces wherever a header-like line
    starts -- this is the fix for the embedding-dilution problem found in
    Stage 4 testing. Without this, a block like "HL7 contact info /
    disclaimer paragraph / 3 SUPPLEMENTAL NOTES / lab code definition" gets
    packed into one chunk purely because it fits under the token budget,
    even though it's visibly three unrelated topics separated by real
    section boundaries in the source document."""
    lines = text.split("\n")
    units: list[str] = []
    current: list[str] = []
    for line in lines:
        if _is_header_line(line) and current:
            units.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        units.append("\n".join(current).strip())
    return [u for u in units if u]


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    """Break a single unit of text down until every piece is <= max_tokens.

    Tries progressively more aggressive separators: sentence boundaries
    first (keeps meaning intact), then a hard character slice as a last
    resort for text with no punctuation to split on (e.g. a run-on table).
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    sentences = re.split(r"(?<=[.!?;])\s+", text)
    if len(sentences) > 1:
        pieces: list[str] = []
        for s in sentences:
            pieces.extend(_split_oversized(s, max_tokens))
        return pieces

    # No sentence boundaries either -- hard-slice by characters.
    max_chars = max_tokens * _CHARS_PER_TOKEN
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def chunk_pages(
    pages: list[tuple[int, str]],
    source_file: str,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 75,
) -> list[Chunk]:
    """Pack page text into overlapping chunks of roughly chunk_size_tokens.

    Why ~500 tokens? It's a deliberate trade-off, not a magic number:
      - Too small (e.g. 50 tokens) -> chunks lose surrounding context, so a
        retrieved chunk like "See section 4.2 for details" is useless on
        its own.
      - Too large (e.g. 4000 tokens) -> each chunk covers many unrelated
        ideas, so embedding similarity gets diluted (the vector represents
        an average of several topics) and you waste context-window space on
        irrelevant text once retrieved.
      - ~300-800 tokens is a common sweet spot for prose/spec documents:
        roughly a few paragraphs, small enough to embed one clear idea per
        chunk, large enough to keep context intact.

    Why overlap at all? Without it, a sentence that straddles a chunk
    boundary gets split, and neither half alone may match a query well.
    ~15% overlap (75/500) means each chunk repeats a bit of the previous
    one's tail, so boundary-straddling ideas still appear intact in at
    least one chunk.

    We chunk at the *page* level (not treating the whole doc as one long
    string) so each chunk keeps accurate page_start/page_end metadata for
    citations, even though that means chunk boundaries respect page breaks
    a little more than a pure sliding window would.
    """
    chunks: list[Chunk] = []
    buffer_text = ""
    buffer_tokens = 0
    buffer_start_page: int | None = None
    buffer_end_page: int | None = None

    def flush():
        nonlocal buffer_text, buffer_tokens, buffer_start_page, buffer_end_page
        if not buffer_text.strip():
            return
        chunks.append(
            Chunk(
                chunk_id=f"{source_file}::chunk-{len(chunks):04d}",
                source_file=source_file,
                page_start=buffer_start_page,
                page_end=buffer_end_page,
                token_count=count_tokens(buffer_text),
                text=buffer_text.strip(),
            )
        )

    for page_num, raw_text in pages:
        text = clean_text(raw_text)
        if not text:
            continue

        # Split each page into paragraph-ish units so we never cut a chunk
        # mid-sentence if we can help it.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        # A spec PDF like this one has dense, un-paragraphed blocks --
        # field tables, segment definitions -- that pypdf hands back as one
        # "paragraph" thousands of tokens long. If we packed those in as-is
        # we'd get oversized chunks that dilute the embedding (see the
        # docstring above). So: recursively fall back to smaller
        # separators, the same idea LangChain's RecursiveCharacterTextSplitter
        # uses -- try paragraph breaks, then sentence breaks, then a hard
        # character cut as a last resort.
        header_split_units: list[str] = []
        for para in paragraphs:
            header_split_units.extend(_split_on_headers(para))

        units: list[str] = []
        for para in header_split_units:
            units.extend(_split_oversized(para, chunk_size_tokens))

        for para in units:
            para_tokens = count_tokens(para)
            starts_with_header = _is_header_line(para.split("\n", 1)[0])

            if buffer_start_page is None:
                buffer_start_page = page_num
            buffer_end_page = page_num

            over_budget = buffer_tokens + para_tokens > chunk_size_tokens
            # Flush on either trigger: size (as before) or a structural
            # section boundary (new) -- a header always starts a fresh
            # chunk, even when there'd be room to keep packing. Exception:
            # if the current buffer is still nearly empty (e.g. we just
            # started a new chunk one header ago and haven't accumulated
            # real body content yet -- two header lines back to back with
            # little between them), don't force a second flush immediately.
            # Otherwise you get degenerate near-empty chunks that are
            # nothing but a header line, which is its own version of the
            # "not enough context to be useful" problem from Stage 1.
            MIN_MEANINGFUL_TOKENS = 20
            header_flush = starts_with_header and buffer_tokens >= MIN_MEANINGFUL_TOKENS
            if buffer_text and (over_budget or header_flush):
                flush()

                if header_flush:
                    # No overlap carried across a structural boundary.
                    # Overlap exists to avoid severing one continuous idea
                    # at an arbitrary size-based cut -- but a header marks
                    # a deliberate topic change, so dragging the previous
                    # section's trailing text forward here would just
                    # reintroduce the dilution problem we're fixing.
                    buffer_text = ""
                    buffer_tokens = 0
                else:
                    buffer_text = _take_last_tokens(buffer_text, overlap_tokens)
                    buffer_tokens = count_tokens(buffer_text)
                buffer_start_page = page_num
                buffer_end_page = page_num

            buffer_text = (buffer_text + "\n\n" + para).strip()
            buffer_tokens = count_tokens(buffer_text)

    flush()
    return chunks


def ingest_pdf(pdf_path: Path, **chunk_kwargs) -> list[Chunk]:
    pages = extract_pages(pdf_path)
    return chunk_pages(pages, source_file=pdf_path.name, **chunk_kwargs)


if __name__ == "__main__":
    # Real filename lives only in the local, gitignored .env as
    # SOURCE_PDF_PATH -- the committed default below is a placeholder so
    # the actual document name never has to appear in tracked source.
    pdf_path = Path(os.environ.get("SOURCE_PDF_PATH", "EDI Specifications/source-spec.pdf"))
    out_path = Path("data/chunks.jsonl")

    chunks = ingest_pdf(pdf_path)

    with out_path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")

    token_counts = [c.token_count for c in chunks]
    print(f"Source: {pdf_path.name}")
    print(f"Chunks produced: {len(chunks)}")
    print(f"Token count  min/avg/max: {min(token_counts)}/{sum(token_counts)//len(token_counts)}/{max(token_counts)}")
    print(f"Wrote: {out_path}")
    print()
    print("=== Sample chunk (#5) ===")
    sample = chunks[5]
    print(f"id={sample.chunk_id}  pages={sample.page_start}-{sample.page_end}  tokens={sample.token_count}")
    print(sample.text[:600])
