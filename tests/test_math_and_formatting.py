"""
Pure formatting/math functions from across both pipelines -- no network,
no filesystem, no mocking needed. Several of these functions are literally
duplicated verbatim across two or three files (to_pgvector_literal exists
in both src/load_vectors.py and src/retrieve.py; to_psycopg3_url exists in
both src_langchain/load_vectors.py and src_langchain/retrieve.py) --
tested once per copy on purpose, since a future edit to one copy and not
the other is exactly the kind of silent drift a test like this catches.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import load_module


# --- cosine_similarity (src/embed.py) ---------------------------------------

embed = load_module("src", "embed")


def test_cosine_similarity_identical_vectors_is_one():
    v = [0.5, 0.3, -0.1, 0.9]
    assert embed.cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-9)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert embed.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0, abs=1e-9)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    assert embed.cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0, abs=1e-9)


# --- to_pgvector_literal (duplicated in load_vectors.py and retrieve.py) ----

load_vectors = load_module("src", "load_vectors")
retrieve = load_module("src", "retrieve")


def test_to_pgvector_literal_format_load_vectors():
    result = load_vectors.to_pgvector_literal([0.1, -0.2, 3.0])
    assert result == "[0.1,-0.2,3.0]"


def test_to_pgvector_literal_format_retrieve():
    # Same function, copy-pasted into a second file -- tested separately
    # so a future edit to only one copy shows up as a test failure, not a
    # silent behavioral drift between the two.
    result = retrieve.to_pgvector_literal([0.1, -0.2, 3.0])
    assert result == "[0.1,-0.2,3.0]"


def test_to_pgvector_literal_empty_vector():
    assert load_vectors.to_pgvector_literal([]) == "[]"


# --- to_psycopg3_url (duplicated in src_langchain/load_vectors.py and retrieve.py) --

lc_load_vectors = load_module("src_langchain", "load_vectors")
lc_retrieve = load_module("src_langchain", "retrieve")


def test_to_psycopg3_url_adds_driver_load_vectors():
    result = lc_load_vectors.to_psycopg3_url("postgresql://user:pw@host:5432/db")
    assert result == "postgresql+psycopg://user:pw@host:5432/db"


def test_to_psycopg3_url_adds_driver_retrieve():
    result = lc_retrieve.to_psycopg3_url("postgresql://user:pw@host:5432/db")
    assert result == "postgresql+psycopg://user:pw@host:5432/db"


def test_to_psycopg3_url_passthrough_for_non_matching_prefix():
    # Defensive case: a URL that's already in the psycopg3 form, or some
    # other scheme entirely, should pass through unchanged rather than
    # getting double-prefixed or mangled.
    already_v3 = "postgresql+psycopg://user:pw@host:5432/db"
    assert lc_load_vectors.to_psycopg3_url(already_v3) == already_v3


# --- build_prompt (src/generate.py) -----------------------------------------

generate = load_module("src", "generate")


def test_build_prompt_single_page_excerpt_label():
    chunks = [{"page_start": 4, "page_end": 4, "content": "Field ZAB-13 is optional."}]
    prompt = generate.build_prompt("What is ZAB-13?", chunks)
    assert "[Excerpt from p. 4]" in prompt
    assert "Field ZAB-13 is optional." in prompt
    assert "Question: What is ZAB-13?" in prompt


def test_build_prompt_multi_page_excerpt_label():
    chunks = [{"page_start": 4, "page_end": 5, "content": "Spans two pages."}]
    prompt = generate.build_prompt("q", chunks)
    assert "[Excerpt from pp. 4-5]" in prompt


def test_build_prompt_joins_multiple_chunks_with_separator():
    chunks = [
        {"page_start": 1, "page_end": 1, "content": "First excerpt."},
        {"page_start": 2, "page_end": 2, "content": "Second excerpt."},
    ]
    prompt = generate.build_prompt("q", chunks)
    assert prompt.index("First excerpt.") < prompt.index("---") < prompt.index("Second excerpt.")


def test_build_prompt_empty_chunks_still_includes_question():
    prompt = generate.build_prompt("orphan question", [])
    assert "Question: orphan question" in prompt


# --- build_context (src_langchain/generate.py) ------------------------------
# Imported via a helper that supplies the fake env vars build_context's own
# module needs at IMPORT time (see conftest.fake_api_env's docstring).

def test_build_context_formats_excerpt_labels(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-fake-voyage-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-fake-anthropic-key")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://fake:fake@localhost:5432/postgres")
    lc_generate = load_module("src_langchain", "generate")

    chunks = [
        {"page": 4, "content": "Field ZAB-13 is optional."},
        {"page": 5, "content": "A second excerpt."},
    ]
    context = lc_generate.build_context(chunks)
    assert "[Excerpt from p. 4]" in context
    assert "[Excerpt from p. 5]" in context
    assert "Field ZAB-13 is optional." in context
