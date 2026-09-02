"""
Tests for _require_env() (duplicated across all 5 src/ files and all 4
src_langchain/ files) and the pacing/retry logic around the embedding
APIs. Nothing here makes a real network call -- retry paths are exercised
against a small fake client class that raises on its first call(s) and
succeeds after, and _pace_requests() is monkeypatched to a no-op so the
suite doesn't actually sleep ~21 seconds per retry test.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import load_module


# --- _require_env, parametrized across every file that defines it -----------
# Copy-pasted per file rather than shared -- see journal/05-testing.md for
# why that's a deliberate, documented choice here, not an oversight. Testing
# each copy separately catches the same "one copy silently drifts from the
# others" risk to_pgvector_literal's duplicate tests catch.

_REQUIRE_ENV_MODULES = [
    ("src", "embed"),
    ("src", "load_vectors"),
    ("src", "retrieve"),
    ("src", "generate"),
    ("src_langchain", "load_vectors"),
    ("src_langchain", "retrieve"),
]


@pytest.mark.parametrize("subdir,modname", _REQUIRE_ENV_MODULES)
def test_require_env_exits_when_var_missing(subdir, modname, monkeypatch, capsys):
    monkeypatch.delenv("SOME_TEST_VAR_THAT_WONT_EXIST", raising=False)
    module = load_module(subdir, modname)

    with pytest.raises(SystemExit) as exc_info:
        module._require_env("SOME_TEST_VAR_THAT_WONT_EXIST")

    assert exc_info.value.code == 1
    assert "SOME_TEST_VAR_THAT_WONT_EXIST" in capsys.readouterr().out


@pytest.mark.parametrize("subdir,modname", _REQUIRE_ENV_MODULES)
def test_require_env_passes_silently_when_var_present(subdir, modname, monkeypatch):
    monkeypatch.setenv("SOME_TEST_VAR_THAT_WONT_EXIST", "present")
    module = load_module(subdir, modname)
    module._require_env("SOME_TEST_VAR_THAT_WONT_EXIST")  # should not raise


@pytest.mark.parametrize("subdir,modname", _REQUIRE_ENV_MODULES)
def test_require_env_reports_only_the_missing_ones(subdir, modname, monkeypatch, capsys):
    monkeypatch.setenv("TEST_VAR_PRESENT", "yes")
    monkeypatch.delenv("TEST_VAR_MISSING", raising=False)
    module = load_module(subdir, modname)

    with pytest.raises(SystemExit):
        module._require_env("TEST_VAR_PRESENT", "TEST_VAR_MISSING")

    out = capsys.readouterr().out
    assert "TEST_VAR_MISSING" in out
    assert "TEST_VAR_PRESENT" not in out


# --- src_langchain/generate.py's import-time check ---------------------------
# This one can't go through the parametrized loop above: it calls
# _require_env at MODULE level (not deferred to a function), specifically
# because it also constructs a real ChatAnthropic(...) client at import
# time -- see that file's own comment for why. So the "missing var" case
# has to be tested via a failed IMPORT, not a direct function call.

def test_langchain_generate_exits_on_import_if_env_missing(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    # The module itself calls load_dotenv(dotenv_path=".env") at import
    # time, and this repo's real (gitignored) .env has real values for
    # exactly these three names. python-dotenv only fills in names NOT
    # already in os.environ, so without this patch it would quietly
    # re-populate the vars we just deleted before _require_env ever runs,
    # and this test would be checking against real credentials instead of
    # the missing-env-var path it's meant to test.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)

    with pytest.raises(SystemExit):
        load_module("src_langchain", "generate")

    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_langchain_generate_imports_fine_with_fake_keys(fake_api_env):
    module = load_module("src_langchain", "generate")
    assert module.MODEL


# --- Pacing / retry logic, mocked client, no real API calls -----------------

class _FailNTimesThenSucceed:
    """Fake voyageai.Client stand-in: raises on the first `fail_count` calls
    to .embed(), then returns a canned result -- one copy of `embedding`
    per text in the batch, matching how the real client always returns
    exactly as many vectors as texts it was given. Records call count so a
    test can assert the retry loop actually retried the right number of
    times."""

    def __init__(self, fail_count, embedding):
        self.fail_count = fail_count
        self.embedding = embedding
        self.calls = 0

    def embed(self, texts, model, input_type):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise RuntimeError("simulated rate limit")
        return _FakeEmbedResult([self.embedding] * len(texts))


class _FakeEmbedResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


def test_embed_batch_with_retry_succeeds_after_transient_failures(monkeypatch):
    embed = load_module("src", "embed")
    monkeypatch.setattr(embed, "_pace_requests", lambda: None)
    monkeypatch.setattr(embed.time, "sleep", lambda seconds: None)

    fake_client = _FailNTimesThenSucceed(fail_count=2, embedding=[0.1, 0.2])
    result = embed._embed_batch_with_retry(fake_client, ["some text"], input_type="document")

    assert result == [[0.1, 0.2]]
    assert fake_client.calls == 3  # 2 failures + 1 success


def test_embed_batch_with_retry_raises_after_exhausting_retries(monkeypatch):
    embed = load_module("src", "embed")
    monkeypatch.setattr(embed, "_pace_requests", lambda: None)
    monkeypatch.setattr(embed.time, "sleep", lambda seconds: None)

    fake_client = _FailNTimesThenSucceed(fail_count=embed.MAX_RETRIES, embedding=[0.1])
    with pytest.raises(RuntimeError):
        embed._embed_batch_with_retry(fake_client, ["text"], input_type="document")
    assert fake_client.calls == embed.MAX_RETRIES


def test_embed_texts_batches_and_paces_every_batch(monkeypatch):
    embed = load_module("src", "embed")
    pace_calls = []
    monkeypatch.setattr(embed, "_pace_requests", lambda: pace_calls.append(1))

    fake_client = _FailNTimesThenSucceed(fail_count=0, embedding=[0.0])
    texts = ["text"] * (embed.BATCH_SIZE * 2 + 3)  # forces 3 batches
    result = embed.embed_texts(fake_client, texts, input_type="document")

    assert len(result) == len(texts)
    assert len(pace_calls) == 3  # one pace call per batch


def test_retrieve_embed_query_retries_on_failure(monkeypatch):
    retrieve = load_module("src", "retrieve")
    monkeypatch.setattr(retrieve, "_pace_requests", lambda: None)
    monkeypatch.setattr(retrieve.time, "sleep", lambda seconds: None)

    class _FakeQueryClient:
        def __init__(self):
            self.calls = 0

        def embed(self, texts, model, input_type):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated rate limit")
            return _FakeEmbedResult([[9.9, 8.8]])

    fake_client = _FakeQueryClient()
    vector = retrieve.embed_query(fake_client, "a question")
    assert vector == [9.9, 8.8]
    assert fake_client.calls == 2


def test_langchain_embed_query_with_retry_same_pattern(monkeypatch):
    lc_retrieve = load_module("src_langchain", "retrieve")
    monkeypatch.setattr(lc_retrieve, "_pace_requests", lambda: None)
    monkeypatch.setattr(lc_retrieve.time, "sleep", lambda seconds: None)

    fake_client = _FailNTimesThenSucceed(fail_count=1, embedding=[1.0, 2.0])
    vector = lc_retrieve._embed_query_with_retry(fake_client, "a question")
    assert vector == [1.0, 2.0]
    assert fake_client.calls == 2
