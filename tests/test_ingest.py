"""
Tests for src/ingest.py -- Stage 1 (manual pipeline).

Everything here is pure text-in, text-out logic: no PDF, no network, no
API key needed. extract_pages() itself (the pypdf wrapper) isn't tested
directly -- it's a thin pass-through to a well-tested third-party library,
and the actual interesting logic (clean_text, header detection, the
size/overlap packing loop) is everything downstream of it, which is what's
covered here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import load_module

ingest = load_module("src", "ingest")


# --- count_tokens ---------------------------------------------------------

def test_count_tokens_uses_four_chars_per_token():
    assert ingest.count_tokens("a" * 40) == 10


def test_count_tokens_never_returns_zero_for_nonempty_text():
    # A single short string still costs "a token" in any real tokenizer --
    # floor(0) would make an empty-looking chunk look free to pack.
    assert ingest.count_tokens("hi") == 1


# --- clean_text -------------------------------------------------------------

def test_clean_text_removes_form_feed():
    assert "\x0c" not in ingest.clean_text("page one\x0cpage two")


def test_clean_text_dehyphenates_wrapped_words():
    # A word hyphenated across a PDF line wrap should rejoin, but only
    # when followed by a lowercase letter -- a real hyphen before a
    # capitalized word (e.g. end of sentence) shouldn't be touched.
    assert ingest.clean_text("cyto-\nlogy result") == "cytology result"


def test_clean_text_collapses_repeated_whitespace_and_blank_lines():
    messy = "field   name\n\n\n\n\nnext section"
    cleaned = ingest.clean_text(messy)
    assert "field name" in cleaned
    assert "\n\n\n" not in cleaned


# --- header detection (the header-flush fix) --------------------------------

def test_header_regex_matches_real_document_header_styles():
    assert ingest._is_header_line("3 SUPPLEMENTAL NOTES")
    assert ingest._is_header_line("6.1.4 Example Segment - Sample ID Fields")


def test_header_regex_does_not_match_bare_subfield_number():
    # A lone "5.2" with its label on the NEXT line must NOT match -- that's
    # specifically what keeps this heuristic from false-triggering on every
    # subfield reference in a dense field table.
    assert not ingest._is_header_line("5.2")


def test_header_regex_does_not_match_ordinary_prose():
    assert not ingest._is_header_line("If the answer is not known, leave this field blank.")


def test_split_on_headers_breaks_at_header_boundary():
    text = (
        "Some disclaimer paragraph about the document license terms.\n"
        "3 SUPPLEMENTAL NOTES\n"
        "This section defines optional fields."
    )
    units = ingest._split_on_headers(text)
    assert len(units) == 2
    assert units[0].startswith("Some disclaimer")
    assert units[1].startswith("3 SUPPLEMENTAL NOTES")


def test_split_on_headers_no_header_present_returns_one_unit():
    text = "Just a normal paragraph with no section heading in it at all."
    assert ingest._split_on_headers(text) == [text]


# --- _split_oversized --------------------------------------------------------

def test_split_oversized_returns_unchanged_when_already_small():
    small = "short text"
    assert ingest._split_oversized(small, max_tokens=500) == [small]


def test_split_oversized_splits_on_sentence_boundaries():
    text = "First sentence here. Second sentence here. Third sentence here."
    # max_tokens small enough to force a split, but text has punctuation,
    # so it should split on sentences rather than hard-cutting characters.
    pieces = ingest._split_oversized(text, max_tokens=5)
    assert len(pieces) > 1
    assert all(ingest.count_tokens(p) <= 5 or " " not in p.strip() for p in pieces)


def test_split_oversized_hard_slices_when_no_punctuation():
    # A run-on table with no sentence punctuation at all -- must fall back
    # to a hard character slice rather than returning one oversized piece.
    text = "x" * 200
    pieces = ingest._split_oversized(text, max_tokens=10)
    assert len(pieces) > 1
    assert all(ingest.count_tokens(p) <= 10 for p in pieces)


# --- chunk_pages: the real bugs this project actually hit -------------------

def test_chunk_pages_produces_expected_chunk_id_and_page_metadata():
    pages = [(1, "A short paragraph of ordinary prose about lab results.")]
    chunks = ingest.chunk_pages(pages, source_file="spec.pdf")
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "spec.pdf::chunk-0000"
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1


def test_chunk_pages_flushes_on_token_budget():
    # Two paragraphs that together exceed a tiny chunk_size_tokens budget
    # must land in separate chunks, not one oversized one.
    long_para_a = "Alpha content word. " * 30
    long_para_b = "Beta content word. " * 30
    pages = [(1, long_para_a + "\n\n" + long_para_b)]
    chunks = ingest.chunk_pages(pages, source_file="spec.pdf", chunk_size_tokens=50, overlap_tokens=10)
    assert len(chunks) >= 2
    assert all(c.token_count <= 60 for c in chunks)  # some slack for the tail overlap carried forward


def test_chunk_pages_header_flush_carries_no_overlap():
    # A header-triggered flush must NOT drag the previous chunk's tail
    # forward -- that would just reintroduce the dilution problem the
    # header-flush fix exists to solve (see journal/01-build-log.md).
    body = "Real substantive content about lab result handling. " * 5  # > MIN_MEANINGFUL_TOKENS
    pages = [(1, body + "\n\n3 SUPPLEMENTAL NOTES\nA new section starts here.")]
    chunks = ingest.chunk_pages(pages, source_file="spec.pdf", chunk_size_tokens=500, overlap_tokens=75)
    assert len(chunks) == 2
    assert chunks[1].text.startswith("3 SUPPLEMENTAL NOTES")
    # No trailing words from chunk 0 leaking into the start of chunk 1.
    assert "lab result handling" not in chunks[1].text


def test_chunk_pages_min_meaningful_tokens_guard_prevents_degenerate_chunk():
    # This is the exact bug found and fixed while building this project:
    # two header lines back-to-back with no real content between them used
    # to force a flush on the FIRST header too, producing a near-empty
    # chunk like "1 OVERVIEW" (2 tokens). The MIN_MEANINGFUL_TOKENS guard
    # means the first header gets absorbed into the still-forming buffer
    # instead, and only the second header (now preceded by real content)
    # triggers a flush.
    pages = [(1, "1 OVERVIEW\n2 SCOPE\nSome real body content follows this second header now.")]
    chunks = ingest.chunk_pages(pages, source_file="spec.pdf", chunk_size_tokens=500, overlap_tokens=75)
    assert all(c.token_count >= 3 for c in chunks), (
        "a degenerate near-empty chunk slipped through -- the "
        "MIN_MEANINGFUL_TOKENS guard should prevent this"
    )


def test_chunk_pages_skips_blank_pages():
    pages = [(1, "   \n\n  "), (2, "Real content on page two.")]
    chunks = ingest.chunk_pages(pages, source_file="spec.pdf")
    assert len(chunks) == 1
    assert chunks[0].page_start == 2
