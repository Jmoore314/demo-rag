"""
Tests for src_langchain/ingest.py -- LangChain rebuild, Stage 1.

ingest_pdf() itself isn't called here -- it goes through PyPDFLoader,
which needs a real PDF file on disk. Instead this constructs
langchain_core.documents.Document objects directly (the same object
PyPDFLoader would hand back) and runs them through the SAME
RecursiveCharacterTextSplitter configuration ingest_pdf() uses, which
exercises the actual chunking behavior without needing a fixture PDF.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import load_module

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

lc_ingest = load_module("src_langchain", "ingest")


def test_count_tokens_matches_manual_pipelines_heuristic_for_ordinary_text():
    assert lc_ingest.count_tokens("a" * 40) == 10


def test_count_tokens_has_no_floor_unlike_manual_pipelines_version():
    # A real, small discrepancy between the two pipelines' otherwise
    # "identical" token heuristics: src/ingest.py's count_tokens() floors
    # at 1 (max(1, len(text)//4)) so an empty/near-empty chunk still costs
    # "a token." This one doesn't -- len("hi")//4 == 0. Harmless in
    # practice (nothing here chunks on a 2-character string), but worth
    # having on record as an actual difference, not an assumption.
    assert lc_ingest.count_tokens("hi") == 0


def _splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=lc_ingest.CHUNK_SIZE,
        chunk_overlap=lc_ingest.CHUNK_OVERLAP,
        length_function=lc_ingest.count_tokens,
    )


def test_splitter_respects_configured_chunk_size():
    # One long page of repeated prose, well over CHUNK_SIZE tokens.
    long_text = "This is a sentence about lab result handling. " * 200
    docs = [Document(page_content=long_text, metadata={"page": 0})]
    chunks = _splitter().split_documents(docs)

    assert len(chunks) > 1
    # Generous slack, not arbitrary -- see the discrepancy test below for
    # exactly why RecursiveCharacterTextSplitter can land meaningfully over
    # the nominal target when given this heuristic as length_function.
    for c in chunks:
        assert lc_ingest.count_tokens(c.page_content) <= lc_ingest.CHUNK_SIZE * 1.6


def test_splitter_can_meaningfully_overshoot_chunk_size_with_this_heuristic():
    """A real finding, not a hypothetical -- caught by this test suite
    itself while writing it, not assumed. RecursiveCharacterTextSplitter's
    _merge_splits() tracks a running total by calling length_function() on
    each SEPARATE small split piece (here, one word at a time when the text
    has no paragraph/line breaks to split on first) and summing those,
    rather than calling length_function() once on the final joined chunk.
    Because count_tokens() is integer division (len(text)//4), that loses
    the remainder on every short fragment -- e.g. a 3-character word scores
    0 "tokens" on its own (3//4==0) even though 200 such words joined by
    spaces are genuinely ~200 tokens once actually measured as one string.
    Verified directly: for 1601 space-separated words, the splitter's own
    incremental sum comes out to 1400, while count_tokens() on the fully
    joined string reports 2300 -- a real ~39% undercount, which is exactly
    why chunks can land well over the nominal 500-token target. This is
    the same category of imprecision as Stage 1's "why the token heuristic
    isn't exact" deep dive in journal/01-build-log.md, just triggered by
    a different mechanism (per-fragment measurement vs. whole-string
    measurement) than that one covers."""
    text = "This is a sentence about lab result handling. " * 200
    words = text.split(" ")

    incremental_sum = sum(lc_ingest.count_tokens(w) for w in words)
    whole_string_count = lc_ingest.count_tokens(" ".join(words))

    assert incremental_sum < whole_string_count * 0.7


def test_splitter_never_produces_a_chunk_spanning_two_page_documents():
    # The documented, deliberate difference from the manual pipeline:
    # PyPDFLoader hands back one Document per page, and split_documents()
    # splits each independently -- so no chunk should mix text from two
    # different `page` metadata values.
    page_a = Document(page_content="Content entirely from page one. " * 20, metadata={"page": 0})
    page_b = Document(page_content="Content entirely from page two. " * 20, metadata={"page": 1})
    chunks = _splitter().split_documents([page_a, page_b])

    pages_seen = {c.metadata["page"] for c in chunks}
    assert pages_seen == {0, 1}
    for c in chunks:
        if c.metadata["page"] == 0:
            assert "page two" not in c.page_content
        else:
            assert "page one" not in c.page_content


def test_splitter_small_document_stays_one_chunk():
    docs = [Document(page_content="A short page.", metadata={"page": 0})]
    chunks = _splitter().split_documents(docs)
    assert len(chunks) == 1
