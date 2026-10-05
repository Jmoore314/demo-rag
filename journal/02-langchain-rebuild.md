# LangChain rebuild -- comparison to the manual pipeline

Direct comparison between the hand-written pipeline in
[`01-build-log.md`](./01-build-log.md) (`src/`) and the LangChain-based
rebuild of the same five stages (`src_langchain/`).

---

## What changed, and why

Rebuilt the same five stages using LangChain (`langchain`,
`langchain-community`, `langchain-text-splitters`, `langchain-postgres`,
`langchain-voyageai`, `langchain-anthropic`), file-for-file against `src/`,
in a new `src_langchain/` directory pointed at a separate Postgres
collection (`demo_rag_langchain`) so it never touches the manual
pipeline's data. Goal wasn't "does LangChain work" -- it was "which parts
of the manual build does it genuinely replace, and which parts still need
the same hand-written fixes regardless."

**What LangChain replaced outright:** `PyPDFLoader` +
`RecursiveCharacterTextSplitter` replaced `extract_pages()`/`clean_text()`/
`chunk_pages()`'s size-based packing loop (same `count_tokens()` heuristic
passed in as `length_function` so chunk-size parameters stayed
apples-to-apples -- 500 target tokens, 75 overlap, on both pipelines).
`VoyageAIEmbeddings` + `PGVector` replaced `embed.py` + `load_vectors.py`'s
hand-written batching and upsert SQL. `similarity_search_with_score_by_
vector()` replaced the hand-written `<=>` SQL in `retrieve.py`.
`ChatPromptTemplate` + `ChatAnthropic` + LCEL (`prompt | model |
StrOutputParser()`) replaced `build_prompt()` + the raw
`anthropic.Anthropic()` call.

**Real findings from reading the installed library source (not
guesses -- checked before writing code around them):**

1. **`VoyageAIEmbeddings.embed_documents()` makes one `tokenize()` API
   call per text, before any embedding call, purely to build batches.**
   Confirmed by reading `_build_batches()` in the installed package. For
   226 chunks that's 226 extra network round-trips with zero pacing or
   meaningful retry between them. On this account's 3 RPM/10K TPM
   throttle, calling it the idiomatic way (`PGVector.from_documents()`)
   would have been slower and more failure-prone than the manual
   pipeline's zero-tokenize-call heuristic (`chars // 4`, no network call
   at all). Worked around it: embedded the bulk corpus with the same
   hand-paced retry logic as `embed.py`, called directly against the raw
   `voyageai` client, then handed the pre-computed vectors to
   `PGVector.add_embeddings()` -- confirmed by reading ITS source too that
   it genuinely performs an `INSERT ... ON CONFLICT (id) DO UPDATE`, the
   same upsert idempotency property as the manual pipeline, not a plain
   insert that would duplicate rows on a re-run.
2. **`VoyageAIEmbeddings.embed_query()` has the identical gap, and it bit
   on the very first real run -- not a hypothetical.** Calling
   `vector_store.similarity_search_with_score()` the idiomatic way threw
   a real `RateLimitError` on the first test question. Root cause:
   `embed_query()` also just calls the client directly, no pacing, and it
   fired immediately after `load_vectors.py`'s bulk run while the
   account's rate-limit window was still hot. The fix needed was *exactly*
   the fix the manual pipeline already made in the same place --
   `src/retrieve.py`'s single query-embed call needed the same pacing as
   `embed.py`'s bulk one, for the same reason (see the Stage 5 section
   below). Rediscovering the identical problem independently in a
   different codebase is good evidence it's a real property of the
   account/library, not a fluke of one pipeline's code.
3. **`ChatAnthropic` handles the `temperature` API removal for you, for
   free.** The manual pipeline hit a real `TypeError` when the current
   Anthropic API rejected the now-removed `temperature` parameter, and
   fixed it by simply not sending the field. `ChatAnthropic`'s
   `temperature` field defaults to `None`, and the wrapper strips every
   `None`-valued field out of the request payload before calling the API
   -- confirmed by reading `_get_request_payload()`. So the natural way to
   use the class (just don't set `temperature`) never hits that error.
   Worth contrasting directly with finding #1 and #2: two integrations
   from the same library, genuinely different levels of built-in
   resilience to the underlying API changing under them or to account-
   level rate limits. Not something you'd know without checking both.
4. **`similarity_search_with_score()`'s "score" is pgvector cosine
   *distance*, not similarity, despite the method name** -- confirmed by
   reading the query-building code (`self.EmbeddingStore.embedding.
   cosine_distance`, results ordered ascending). Silently backwards if
   misread: both distance and similarity are floats in a similar numeric
   range, so treating "closest" as "best" requires knowing which
   convention is in play. `1 - score` converts it to the same
   `similarity` number the manual pipeline reports, so both pipelines'
   outputs are directly comparable below.

**Chunking difference, confirmed in the numbers:** `src_langchain/
ingest.py` produced 226 chunks against the manual pipeline's 216 from the
identical size target. Root cause, predicted before running it:
`PyPDFLoader` loads one `Document` per page and `split_documents()`
splits each page's Document independently, so no chunk can span a page
break here, unlike the manual pipeline's whole-document text stream. A
paragraph straddling a page boundary in the source gets cut there
regardless of size budget -- more, slightly smaller chunks as a direct,
measurable consequence of that one design difference.

**Retrieval comparison (same 3 test questions as Stage 4):**

- *PID segment question:* near-identical result to the manual pipeline's
  post-header-flush-fix version -- all three genuine occurrences of the
  PID segment definition retrieved cleanly in the top 3, similarities in
  the same range (this pipeline: 0.688/0.685/0.682; manual: 0.701/0.700/
  0.699). This directly **contradicts a prediction made before running
  it** (see `src_langchain/ingest.py`'s docstring): expected this
  pipeline to reproduce the same dilution problem the header-flush fix
  solved, since `RecursiveCharacterTextSplitter` has no header-detection
  logic at all. It didn't, because of an interaction between two design
  differences that hadn't been considered together: each of the three
  PID-segment occurrences happens to start right at the top of a page in
  the source document, so the page-boundary restriction (adopted for an
  unrelated reason -- no cross-page chunks) incidentally produces the
  same clean-boundary effect the header-flush fix was purpose-built for.
  Good reminder that predicting a multi-variable system by reasoning
  about one variable at a time can miss real interactions -- worth
  stating the wrong prediction plainly rather than quietly revising it
  after the fact.
- *Cytology question:* same known failure mode as the manual pipeline --
  adjacent, near-identical field templates score closely together, making
  the single correct field hard to distinguish by similarity alone (top
  two results here: 0.561 vs. 0.557, essentially tied). Confirms this is
  a genuine property of the document's heavily-templated field tables,
  not an artifact of one particular chunking strategy -- neither
  pipeline's chunker addresses it, both show the same symptom.
- *Lab code question:* the standout result -- 0.580 here vs. 0.500 in the
  manual pipeline, and from a different occurrence of the concept than
  the manual pipeline retrieved. The document defines "lab code" in more
  than one place (same multi-occurrence pattern as the PID segment); the
  manual pipeline's best available chunk for this question sits in a
  section that mixes several unrelated topics under one heading with no
  further sub-heading to split on (documented as an accepted limitation
  in the header-flush-fix section above) -- a genuine structural property
  of that specific spot in the document, not fixable by chunking strategy
  alone. This pipeline's page-bounded chunker happened to isolate a
  *different* occurrence of the same definition that sits alone on its
  page, without that contamination. The honest lesson isn't "LangChain
  chunks better" -- it's that when a document defines the same concept in
  multiple places, which chunking strategy you use can determine which
  occurrence retrieval actually finds, and quality can differ for reasons
  specific to where in the document each occurrence happens to sit.

**Generation comparison (same 4 test questions as Stage 5, including the
France refusal test):** all four questions produced correctly grounded
answers, verified against the retrieved-chunk previews already inspected
during the retrieval comparison above.
- The cytology answer was broader than the manual pipeline's -- it named
  8 fields in the affected family rather than 6, because `top_k=5` here
  happened to retrieve five adjacent pages covering more of the field
  range. A real example of the Stage 5 noise-tolerance finding recurring
  in a second, independently-built pipeline: retrieval doesn't need to be
  precise for generation to produce a complete, correct answer.
- The PID segment answer matched the manual pipeline's key grounding
  signal exactly: it distinguished required fields from PID-3/PID-4
  (marked optional in this specific document) rather than defaulting to
  generic HL7 convention -- the same evidence of genuine grounding over
  general training knowledge, reproduced independently in a second
  pipeline.
- The out-of-scope question was correctly refused, with the same
  low-similarity signal (best score ~0.19) seen in the manual pipeline's
  equivalent test (~0.16-0.18) -- consistent evidence that a
  similarity-score gap is a real, reusable out-of-scope signal, not
  specific to one pipeline's numbers.

**Overall verdict:** LangChain replaced real code (loader, splitter,
batching/upsert SQL, hand-written retrieval SQL, prompt formatting) with
library calls, and in one place (`ChatAnthropic`'s handling of the
`temperature` removal) it actively protected against an API change that
cost real debugging time in the manual build. But it did not remove the
need to understand what's actually happening underneath: the exact same
rate-limit pacing problem had to be diagnosed and fixed by hand in both
pipelines, in the same place, for the same reason -- and a prediction
made purely by reading two pieces of code in isolation (no header-flush
fix implies dilution) turned out wrong once the two pipelines' actual
behavior interacted with the real document. The honest summary for an
interview: a framework can hand you the common-case plumbing, but the
moment your account has a rate limit, or your document has real-world
quirks, or two design choices interact in a way the docs never mention,
you're back to reading source code and reasoning about your specific
system either way. Having built both versions is worth more than either
one alone.

---

## Post-reseed check: a retrieval gap in the LangChain pipeline (Oct 5, 2026)

The Supabase project had been paused and its tables came back empty, so
both pipelines were reloaded from the saved chunk files and run again
through their `generate.py` scripts with the same four test questions.
Both worked end to end. Three of the four questions produced equivalent,
correctly cited answers (the cytology question, the lab code question,
and the out-of-scope France question, which both pipelines refused with
top similarity scores around 0.2). The fourth, the PID segment question,
exposed a real difference.

**What happened.** The manual pipeline listed PID-8 (Patient Gender) among
the required PID fields. The LangChain pipeline left it out, and its
answer gave no sign that anything was missing. Checking the source PDF
confirmed the manual pipeline was right: page 19 marks PID-8 as required.

**Why.** The LangChain splitter cut page 19's field table into two chunks.
`lc-chunk-0030` (525 tokens) holds the table down through PID-7, and
`lc-chunk-0031` (145 tokens) holds PID-8. The top 5 results for the
question included chunk 0030 but not chunk 0031, so the model never saw
PID-8 and had no way to know it was absent. The manual pipeline avoided
this by chance: its chunks span page boundaries, and one of its top 5 was
a different page (pp. 122-123) that also listed PID-8.

**Why it matters.** This is a retrieval coverage problem, not a generation
problem. The model followed its grounding rules correctly and answered
only from what it was given. A grounded answer is only as complete as the
excerpts retrieved, and a table that gets split across chunks is a
natural place for completeness to break. It also showed up as a confident,
cleanly formatted answer, which is harder to catch than a refusal.

**Possible fixes (none tried yet).**
- Raise `top_k` from 5 to something like 7 for the LangChain pipeline.
  This would probably pull in the missing sibling chunk for this
  question, but it is a guess until tested, and it costs more prompt
  tokens on every question.
- Keep small trailing fragments attached to their neighbor, for example
  by merging a chunk under a minimum size back into the previous chunk
  from the same page.
- Include adjacent chunks from the same page whenever one chunk from that
  page is retrieved.

This is also an example of the kind of check that a unit test cannot
give, as described in [`05-testing.md`](./05-testing.md). Whether the
retrieved set is complete for a given question needs a small evaluation
set with known correct answers, not just mocked plumbing.

---

## Key terms — LangChain rebuild

- **LCEL (LangChain Expression Language):** the `|` (pipe) operator for
  composing a prompt, a model, and an output parser into one callable
  "chain" (`prompt | model | StrOutputParser()`), each stage's output
  automatically becoming the next stage's input.
- **Runnable:** the common interface LCEL components implement
  (`.invoke()`, plus streaming/batch variants) so arbitrarily different
  pieces -- prompts, models, parsers, retrievers -- can be composed with
  the same `|` syntax regardless of what each one actually does
  internally.
- **Embeddings interface / VectorStore interface:** LangChain's abstract
  base classes for "anything that turns text into vectors" and "anything
  that stores and searches vectors." The reason swapping Voyage for a
  different embedding provider, or pgvector for a different vector
  database, mostly means changing one constructor call rather than
  rewriting pipeline logic. The tradeoff for that swappability: the
  abstraction can hide provider-specific details (like Voyage's per-text
  tokenize-call batching, or which distance metric a "score" actually is)
  that matter a lot once you're rate-limited or need exact numbers.
- **Output parser:** the LCEL component that converts a model's raw
  response object into a simpler shape -- here, `StrOutputParser()`
  extracts just the plain text answer, equivalent to
  `response.content[0].text` in the manual pipeline.

