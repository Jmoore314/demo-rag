# Testing

Where this fits: a `pytest` suite covering what's reasonable to test
without spending API money or needing a live database, plus the reasoning
for why the rest of "does this pipeline actually work well" isn't a unit
test at all. 67 tests, `pytest tests/`, zero network calls, zero cost.

---

## Two kinds of "does this work," and why only one of them is a unit test

Talked this through before writing any test code, and it's worth writing
down because it's the actual design decision behind everything in
`tests/`, not just a scoping note.

A RAG pipeline splits cleanly into deterministic software and
non-deterministic model calls, and they need genuinely different kinds of
checking:

- **The deterministic parts** -- chunking logic, URL/vector-literal
  formatting, prompt construction, retry/backoff behavior -- are pure or
  near-pure functions. Given the same input, they produce the same output
  every time, so classic unit testing (and TDD, if these were being built
  test-first) applies to them exactly the way it applies to any other
  software. This is what `tests/` actually covers.
- **Retrieval and generation quality** -- does the right chunk come back
  for a question, does the generated answer stay grounded -- can't be unit
  tested the same way, because the same prompt can produce different
  wording from the same model on different runs. Asserting exact output
  text would be either trivially mocked (proving nothing) or genuinely
  flaky. What actually measures this is an **evaluation harness**:
  precision@k / recall@k against a labeled question set, faithfulness
  scoring (RAGAS-style), thresholds instead of exact-match assertions, run
  periodically rather than on every change because it costs real API
  money and takes real time. This project's `test_questions` lists at the
  bottom of each script's `__main__` are the closest thing to that here --
  manual, eye-verified, and honestly labeled as such rather than dressed
  up as automated tests. A real eval harness (a labeled question set,
  RAGAS or a hand-rolled scorer, a regression threshold) is the natural
  next step if this pipeline needed to prove its retrieval quality rather
  than just demonstrate it -- not built here, on purpose, since it needs
  real API calls to mean anything, which conflicts directly with "keep
  this suite free to run."

So: `pytest` for "is the plumbing correct," an eval harness (not built
here) for "are the answers good." Conflating the two -- unit-testing
model output, or treating a `test_questions` printout as a real test
suite -- is the mistake this split is meant to avoid.

---

## A real structural problem this suite had to work around

Every script in `src/` and `src_langchain/` uses **bare sibling
imports** -- `src/generate.py` does `from retrieve import retrieve`, not
`from .retrieve import retrieve`. That's a deliberate, simple pattern for
scripts meant to be run directly (`python3 src/generate.py`), where
Python automatically puts the script's own directory on `sys.path`. It
is NOT package-shaped, though, and that caused two real problems the
moment tests tried to import these files instead of running them:

1. `src/generate.py`'s own `from retrieve import retrieve` only resolves
   if `src/` is on `sys.path` at import time -- a plain `import
   src.generate` from a test at the repo root doesn't provide that.
2. `src/retrieve.py` and `src_langchain/retrieve.py` are both,
   unavoidably, a module literally named `retrieve`. If both directories
   ever ended up on `sys.path` in the same process (which a single
   `pytest` run touching both pipelines requires), Python's module cache
   would silently hand back whichever one got imported first -- a test
   claiming to test the LangChain version could silently be running
   against the manual version's code instead, with no error.

Fixed with a small loader in `tests/conftest.py` (`load_module(subdir,
modname)`) that does exactly what running the script directly would do:
temporarily prepend the target file's own directory to `sys.path`, evict
any previously-cached copy of that module name from `sys.modules` first,
import, then take the directory back off `sys.path`. Every test file
imports the module under test through this loader rather than a plain
`import` statement -- verified directly (not assumed) by importing
`retrieve` from `src/`, then from `src_langchain/`, then from `src/`
again in the same process and asserting each one resolves to the correct
file.

The honest long-term fix is converting both directories into real
packages with relative imports. Not done here -- it would mean touching
every file in both pipelines for a benefit (importability) that only
testing needs; running the scripts directly, which is the repo's actual
primary use case, doesn't care either way.

---

## Two real bugs the test suite itself caught, not hypotheticals

Same spirit as the rest of this project: verify, don't assume. Both of
these were found *while writing the tests*, not planned in advance.

**1. `python-dotenv` silently undoes a test's env-var deletion.** Every
script calls `load_dotenv(dotenv_path=".env")` at import time, and this
repo's real (gitignored) `.env` has real values for `VOYAGE_API_KEY`,
`ANTHROPIC_API_KEY`, and `SUPABASE_DB_URL`. `load_dotenv()` only fills in
names that are NOT already present in `os.environ` -- so a test that
deletes one of those three specific names via `monkeypatch.delenv(...)`
and then imports the module gets it silently repopulated from the real
`.env` file before `_require_env()` ever runs, because from
`load_dotenv()`'s point of view the name is now "missing" and fair game
to fill in. The test meant to prove the friendly-error-message feature
actually works would have silently passed against real credentials
instead of the missing-var path it claimed to test. Fixed by also
patching `dotenv.load_dotenv` to a no-op for that one test. This only
bites tests that delete one of the three *real* `.env` key names --
tests using a synthetic name like `SOME_TEST_VAR_THAT_WONT_EXIST` were
never at risk, since `load_dotenv()` has nothing to restore for a name
that was never in the `.env` file to begin with.

**2. `RecursiveCharacterTextSplitter` can meaningfully overshoot
`chunk_size` with this project's token heuristic specifically.** Writing
a test asserting LangChain's splitter respects `CHUNK_SIZE` (500) against
a long block of plain prose, one chunk came back at 719 tokens -- a 44%
overshoot, not noise. Traced to the actual library source (same
methodology as every other finding in this journal): the splitter's
`_merge_splits()` measures length by calling `length_function()` on each
small split piece separately (here, one word at a time, since the test
text had no paragraph or line breaks to split on first) and summing
those -- rather than calling it once on the final joined chunk. Because
`count_tokens()` is integer division (`len(text)//4`), that loses the
remainder on every short fragment: a 3-character word scores 0 "tokens"
on its own (`3//4==0`), even though hundreds of such words joined by
spaces are genuinely hundreds of tokens once measured as one string.
Proved directly, not just asserted: for 1,601 space-separated words, the
splitter's own incremental sum comes out to 1,400, while `count_tokens()`
on the fully joined string reports 2,300 -- a real ~39% undercount. This
is the same *category* of imprecision as the "why the token heuristic
isn't exact" deep dive in
[`01-build-log.md`](./01-build-log.md) (Stage 1), just triggered by a
different mechanism -- per-fragment measurement compounding rounding
loss, rather than that deep dive's character-density argument. Handled
by widening the test's tolerance to a real, documented multiple rather
than an arbitrary one, plus adding a dedicated test that proves the
discrepancy directly instead of quietly working around it.

---

## What's covered, file by file

| Test file | Covers | Real bugs/findings it surfaced |
|---|---|---|
| `test_ingest.py` | `src/ingest.py`: `count_tokens`, `clean_text`, header detection, `_split_on_headers`, `_split_oversized`, `chunk_pages` | Directly re-proves the header-flush fix and the `MIN_MEANINGFUL_TOKENS` degenerate-chunk guard from Stage 1 -- both real bugs already documented there, now regression-protected |
| `test_langchain_ingest.py` | `src_langchain/ingest.py`: `count_tokens`, the configured `RecursiveCharacterTextSplitter` (via constructed `Document` objects, no PDF needed) | The `count_tokens` floor discrepancy (finding #2, above) and the chunk-size-overshoot discrepancy |
| `test_math_and_formatting.py` | `cosine_similarity`, both copies of `to_pgvector_literal`, both copies of `to_psycopg3_url`, `build_prompt`, `build_context` | Each duplicated helper tested separately on purpose -- a future edit to one copy and not its twin shows up as a test failure, not silent drift |
| `test_retry_and_env.py` | `_require_env` across all 6 files that define it (parametrized), `src_langchain/generate.py`'s import-time check, the pacing/retry logic in `embed.py`, `retrieve.py`, and `src_langchain/retrieve.py` (all against a fake client that fails N times then succeeds) | The `python-dotenv` re-population gotcha (finding #1, above) |
| `test_pipeline_plumbing.py` | `retrieve()` against a mocked `voyageai.Client` + mocked `psycopg2.connect`; `generate_answer()` (both pipelines) against a mocked retrieval + mocked LLM client | Confirms the pgvector distance-to-similarity conversion and that the retrieved content actually lands in the prompt sent to the model -- the two places a silent wiring bug would be easy to miss |

Not covered, on purpose: `extract_pages()` (a thin wrapper over `pypdf`,
low value to test against a well-tested third-party library), and
anything requiring a real PDF, a real Postgres connection, or a real API
call -- all deliberately mocked out instead, per the split at the top of
this file.

---

## Running it

```bash
pip install -r requirements-dev.txt
pytest
```

`requirements-dev.txt` is kept separate from `requirements.txt` on
purpose -- nothing that clones this repo to actually *run* the pipeline
needs `pytest`; only someone running the test suite does.
