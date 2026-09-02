"""
"Does the wiring work" tests for the two functions that actually reach out
to an API and a database in one call: retrieve() (embeds a query, queries
Postgres) and generate_answer() (calls retrieve(), then calls Claude).
Both the DB connection and the API clients are mocked -- these tests check
that the right SQL gets built and the right data flows through in the
right shape, not that a real vector index returns good results (that's
what the eval-style question sets in each script's own __main__ are for,
and why they're not duplicated here -- see journal/05-testing.md).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import load_module


# --- src/retrieve.py: retrieve() ---------------------------------------------

def test_retrieve_assembles_results_from_mocked_db_and_api(monkeypatch):
    retrieve = load_module("src", "retrieve")

    # Fake the embedding call so it never touches Voyage.
    fake_voyage_client = MagicMock()
    fake_voyage_client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2, 0.3]])
    monkeypatch.setattr(retrieve.voyageai, "Client", lambda api_key: fake_voyage_client)
    monkeypatch.setattr(retrieve, "_pace_requests", lambda: None)

    # Fake psycopg2.connect(...) as a context-manager chain returning one
    # canned row, shaped exactly like the real cursor.fetchall() would.
    fake_row = ("spec.pdf::chunk-0080", "spec.pdf", 59, 60, "Field ZAB-13 text.", 0.15)
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = [fake_row]
    fake_cursor.__enter__.return_value = fake_cursor
    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor
    fake_conn.__enter__.return_value = fake_conn
    monkeypatch.setattr(retrieve.psycopg2, "connect", lambda url: fake_conn)
    monkeypatch.setenv("VOYAGE_API_KEY", "fake")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://fake/fake")

    results = retrieve.retrieve("What is ZAB-13?", top_k=3)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "spec.pdf::chunk-0080"
    assert results[0]["page_start"] == 59
    # pgvector's <=> is distance; retrieve() must convert to similarity.
    assert results[0]["similarity"] == pytest.approx(0.85)

    # The query embedding actually went into the SQL params, not silently
    # dropped or hardcoded.
    called_sql, called_params = fake_cursor.execute.call_args[0]
    assert called_params["top_k"] == 3
    assert "0.1" in called_params["query_vector"]


# --- src/generate.py: generate_answer() --------------------------------------

def test_generate_answer_grounds_prompt_in_retrieved_chunks(monkeypatch):
    generate = load_module("src", "generate")

    fake_chunks = [
        {"chunk_id": "c1", "page_start": 4, "page_end": 4, "content": "Field ZAB-13 is optional.", "similarity": 0.9}
    ]
    monkeypatch.setattr(generate, "retrieve", lambda question, top_k: fake_chunks)

    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="ZAB-13 may be left blank (p. 4).")]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setattr(generate.anthropic, "Anthropic", lambda **kwargs: fake_client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    result = generate.generate_answer("What is ZAB-13?", top_k=5)

    assert result["answer"] == "ZAB-13 may be left blank (p. 4)."
    assert result["chunks_used"] == fake_chunks

    # The actual prompt sent to Claude must contain the retrieved content
    # and be grounded via the system prompt -- this is the whole point of
    # the RAG pattern, not an incidental detail.
    _, call_kwargs = fake_client.messages.create.call_args
    assert "Field ZAB-13 is optional." in call_kwargs["messages"][0]["content"]
    assert call_kwargs["system"] == generate.SYSTEM_PROMPT


def test_generate_answer_omits_workspace_header_when_not_set(monkeypatch):
    generate = load_module("src", "generate")
    monkeypatch.setattr(generate, "retrieve", lambda question, top_k: [])
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    captured_kwargs = {}

    def fake_anthropic(**kwargs):
        captured_kwargs.update(kwargs)
        client = MagicMock()
        client.messages.create.return_value = MagicMock(content=[MagicMock(text="ok")])
        return client

    monkeypatch.setattr(generate.anthropic, "Anthropic", fake_anthropic)
    generate.generate_answer("q")

    assert captured_kwargs["default_headers"] is None


# --- src_langchain/generate.py: generate_answer() ----------------------------

def test_langchain_generate_answer_grounds_prompt_in_retrieved_chunks(fake_api_env, monkeypatch):
    lc_generate = load_module("src_langchain", "generate")

    fake_chunks = [{"chunk_id": "lc-chunk-0080", "page": 59, "content": "Field ZAB-13 text.", "similarity": 0.85}]
    monkeypatch.setattr(lc_generate, "retrieve", lambda vector_store, question, top_k: fake_chunks)
    # `chain` is a real LangChain RunnableSequence (a pydantic model), which
    # rejects setting an attribute that isn't one of its declared fields --
    # patching .invoke directly on it raises a pydantic ValueError. Replace
    # the whole module-level `chain` name with a stand-in object instead.
    monkeypatch.setattr(
        lc_generate, "chain", MagicMock(invoke=lambda inputs: "ZAB-13 may be left blank (p. 59).")
    )

    result = lc_generate.generate_answer(vector_store=object(), question="What is ZAB-13?", top_k=5)

    assert result["answer"] == "ZAB-13 may be left blank (p. 59)."
    assert result["chunks_used"] == fake_chunks
