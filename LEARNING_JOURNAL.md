# RAG from Scratch — Learning Journal

A running record of what I built, why I built it that way, and the vocabulary
that goes with it. Source document: a 140-page HL7 lab-interface technical
spec PDF (excluded from this repo -- not mine to publish; see README).
Goal: build a working RAG pipeline by hand first, then rebuild it with
LangChain to see what the framework buys you.

**Architecture chosen:** Voyage AI for embeddings → Supabase Postgres
(pgvector) for storage/retrieval → Claude (Anthropic API) for generation.

---

## Progress

- [x] Stage 1 — Document ingestion & chunking
- [x] Stage 2 — Embeddings (Voyage AI)
- [x] Stage 3 — Vector storage (Supabase + pgvector)
- [x] Stage 4 — Retrieval logic
- [x] Stage 5 — Prompt construction & Claude generation
- [x] Verification pass (folded into Stage 5 testing -- see below)
- [x] LangChain rebuild

**Rough time estimate for the full project (manual build + LangChain
rebuild):** ~5–9 hours of hands-on, explained work, most realistically spread
across a few sessions rather than one sitting. Stage 1 (including debugging
the chunk-size and tiktoken issues below) took roughly 45 minutes. Supabase
setup in Stage 3 will likely be the next-longest stage because of project
provisioning time.

---

## LangChain rebuild — comparison to the manual pipeline

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

## Backlog decision — scoping the chunking refinement (not yet implemented)

Before deciding whether to act on the two Stage 4 findings, worked out
exactly what "fixing" each one would mean.

**Embedding dilution (chunk-0007) -- scoped fix, worth doing eventually:**
Root cause is precise: `chunk_pages()` in `ingest.py` only ever checks
token count before deciding to keep packing paragraphs into the current
chunk buffer -- it has no concept of topic or document structure. The fix
is narrow and surgical: detect section-header-like lines (pattern: leading
number + heading, e.g. "3 SUPPLEMENTAL NOTES", "6.1.4 Example Segment -
Sample Identification Fields" -- both real patterns in this document) and force a
buffer flush whenever one appears, even under the token budget, so a new
chunk always starts fresh at a structural boundary instead of potentially
absorbing trailing unrelated content from the prior section. Small,
bounded change (one regex check + an early flush call). Caveat: this
heuristic is tuned to this document's specific header formatting and
wouldn't automatically generalize to a differently-formatted document --
the more general fix would be "semantic chunking" (detecting topic shifts
via sentence-level embedding comparison instead of text pattern-matching),
a meaningfully bigger lift not currently justified.

**Boilerplate/template collision (ZCY fields) -- deliberately NOT fixing,
with reasoning:** A real fix means detecting and stripping repeated
templated text before embedding (while keeping it in what's shown to the
LLM), which needs reasonably robust template-detection logic with real
risk of stripping meaningful content that resembles the pattern. More
importantly, Stage 5 already showed the overall pipeline is tolerant of
this specific imperfection -- even with 2 borderline-irrelevant chunks in
the retrieved set, Claude's answer correctly used only the genuinely
relevant ZAB-10 through ZAB-15 fields (verified against source) and wasn't
thrown off by the noise. Generalizable lesson: retrieval doesn't have to
be perfect for the overall system to produce a correct answer, because a
properly grounded generation step has some built-in resilience to noisy
context. Not worth solving a problem that isn't currently costing anything.

**Decision:** if/when revisited, scope the work to the header-flush fix
only. Leave the boilerplate issue as an accepted, documented limitation.

---

## Chunking refinement — implementing the header-flush fix

Followed through on the backlog decision above. Changes made to
`ingest.py`:

**1. Header detection.** Added `_HEADER_RE`, a regex matching lines that
look like this document's section headings -- a numeric prefix (`5.3.2`,
`2`, etc.) followed by a short Title-Case-ish phrase (`^\d+(\.\d+)*\s+[A-Z]
[A-Za-z0-9 ,\-/&]{2,80}$`). `_is_header_line()` wraps it. Deliberately
pattern-based rather than ML-based -- this is a heuristic tuned to *this*
document, not a general solution (documented as a known limitation above).

**2. Header-boundary flushing.** In the chunk-packing loop, before packing
each new paragraph, check whether it *starts* with a header line
(`starts_with_header`). If so, and the current buffer already holds at
least `MIN_MEANINGFUL_TOKENS = 20` tokens of real content, flush the
buffer immediately (`header_flush`) before starting the new paragraph, and
carry **no overlap** forward into the new chunk -- unlike a normal
token-budget flush, a header-triggered flush is a deliberate topic
boundary, so bleeding the end of the old section into the start of the
new one would reintroduce the exact dilution problem this fix targets.

**3. The `MIN_MEANINGFUL_TOKENS` guard -- a bug found on the first run.**
The first version flushed on *every* header line unconditionally, which
produced a degenerate chunk: `chunk-0007 = "1 OVERVIEW"` (2 tokens) --
two headers appeared back-to-back with no real content between them, so
the first header's forced flush emitted a near-empty chunk. A chunk with
2 tokens of content is worse than the dilution problem it was meant to
fix -- it wastes a retrieval slot and can't answer anything on its own.
Fix: only force a flush if the buffer already has >= 20 tokens of
substance; otherwise let the header line get absorbed into the buffer
that's still forming, and only flush at the *next* header or token limit.
Re-running after this guard brought the chunk count from 217 down to 216
and the minimum chunk size from 2 tokens up to 27 -- no more degenerate
chunks.

**4. Downstream consequence -- `load_vectors.py` needed a `TRUNCATE`.**
The original loader used `ON CONFLICT (chunk_id) DO UPDATE` (an upsert),
which is correct for re-running the *same* chunking (e.g. after fixing a
typo in the prompt) but silently wrong for a re-*chunking*: chunk
boundaries shifted, so `chunk_id`s like `chunk-0080` now point to
completely different text than before. An upsert would have left 216 new
rows correctly updated but any leftover rows from the old 194/217-chunk
runs stranded in the table under stale IDs. Added `TRUNCATE
document_chunks` immediately before the upsert to guarantee a clean
full reload whenever the chunking itself changes.

**Before/after verification -- re-ran all three Stage 4 test questions
after re-embedding and re-loading the 216 new chunks:**

- **Q1 (cytology, ZCY fields) -- essentially unchanged** (top similarity
  0.5798 -> 0.5783, correct chunk still ranked #2). Expected: this
  question's problem was boilerplate/template collision, a different
  failure mode this fix doesn't target (see the backlog decision above --
  deliberately not fixed).
- **Q3 (PID segment) -- clearly improved.** Before: only 1 of the top 3
  results was a genuine PID-segment chunk. After: all 3 top results
  (chunk-0184 at 0.7010, chunk-0038 at 0.7001, chunk-0135 at 0.6993) are
  genuine PID-segment content. Root cause of the improvement: this
  document defines the PID segment **three separate times** (likely once
  in an overview section and twice in per-message-type detail sections). Before the fix, at least one of
  those three occurrences was chunked together with trailing unrelated
  text, diluting its embedding enough to rank below an off-topic chunk.
  The header-flush fix gives each of the three occurrences a clean,
  undiluted chunk, so all three now legitimately cluster near the top for
  a PID-segment query. This is a bigger win than expected going in --
  the original hypothesis was "cleaner chunks," not "consistently clean
  across multiple occurrences of the same topic."
- **Q2 (lab code) -- barely moved despite being genuinely cleaner**
  (0.4980 -> 0.4998). Investigated why: pulled the full text of the
  current lab-code chunk (`chunk-0011`, 420 tokens). It's cleaner than
  before -- the old front-matter/disclaimer contamination is gone -- but
  it *still* mixes three distinct things under one document heading ("3
  SUPPLEMENTAL NOTES") that has no further sub-heading underneath it: (1)
  the prose definition of a lab code, (2) a raw pipe-delimited example HL7
  message (a mix of standard HL7 segments plus a custom Z-segment), and
  (3) an unrelated provider-ID note referencing two other fields. A header-based
  heuristic can only split at boundaries the document itself marks with a
  heading -- it has no way to detect a topic shift *within* a single
  un-subdivided section. This is an honest, bounded result: the fix
  removed the worst *external* contamination (content bleeding in from
  neighboring sections) but can't fix *internal* heterogeneity within a
  section the source document itself didn't subdivide. Fixing this
  further would require actual semantic chunking (splitting on detected
  topic shifts within a paragraph run, not just on header lines) --
  correctly identified back in the backlog decision as the "meaningfully
  bigger lift not currently justified."
- **A caveat about `embed.py`'s built-in sanity check, so the numbers it
  prints aren't misread as a regression:** that script compares fixed
  array indices (chunk 80 vs 81, chunk 2 vs 80) and printed
  0.8598 -> 0.7835 and 0.6146 -> 0.619 after re-chunking, which looks like
  embeddings got *less* similar. Checked directly: re-chunking shifted
  every chunk boundary, so those index numbers now point to entirely
  different text than they did before (old index 80/81 was ZCY fields on
  pages 59-60; new index 80/81 is lab-result-value codes on page 49-50
  and a date/time field on page 50). The apparent drop is an artifact of
  comparing unrelated content at the same array position across two
  different chunkings, not a real similarity decline. Lesson: a "compare
  chunk N to chunk N+1" sanity check is only meaningful *within* one
  fixed chunking, never across two different chunkings of the same
  document -- something worth remembering before trusting any positional
  index as a stable identity.

**Net assessment:** the fix did exactly what it was scoped to do (fix
dilution at document-marked structural boundaries) and no more (it
can't fix boilerplate collision or intra-section heterogeneity, both
already known and accepted limitations). Real, measurable retrieval
improvement on 1 of 3 test questions, neutral on the other 2 for reasons
that were predictable going in.

---

## Stage 5 — Prompt construction & Claude generation

**File:** `src/generate.py`

**What it does:** `generate_answer(question, top_k=5)` calls `retrieve()`
from Stage 4, formats the chunks as page-labeled excerpts, sends them plus
the question to Claude with a grounding-focused system prompt, and returns
the answer alongside which chunks were used.

**Key decisions:**

- **The system prompt is the entire mechanism that makes this RAG instead
  of a chatbot with extra text pasted above the question.** Explicit
  instructions: answer only from provided excerpts, refuse (don't guess)
  if insufficient, cite page numbers for every claim. Without this, a
  capable model will happily blend retrieved context with its own general
  training knowledge.
- **Page citations are baked into the excerpt text itself** (`[Excerpt
  from p. X]`), not left as metadata Claude never sees -- it can only cite
  a page it was actually shown in-line.
- **Deliberately included an out-of-scope test question** ("What is the
  capital of France?") -- the real test of grounding isn't whether good
  questions get good answers, it's whether irrelevant ones get correctly
  refused instead of answered from general knowledge.

**Real API surprises while building this** (current as of Sept 2026, all
genuinely new since my training data, not oversights):
1. `temperature` is no longer a Messages API parameter at all -- removed
   in favor of `output_config.effort` (controls reasoning *depth*, not
   randomness -- a different concept, not a rename). No sampling-based
   determinism knob exists in this API version.
2. `effort` is only supported on some model generations, not
   `claude-sonnet-4-5` -- got a 400 "this model does not support the
   effort parameter" and just dropped output_config entirely rather than
   chase per-model compatibility.
3. "Identity-linked" API keys (tied to a person, not a workspace) require
   an `anthropic-workspace-id` header unless scoped to one workspace at
   creation. Fixed by creating a workspace-scoped key.
4. The Claude API requires a funded billing balance before any request
   succeeds at all -- unlike Voyage's throttled-but-working free tier,
   there's no reduced-limits fallback here.
5. `retrieve.py`'s single query-embed call needed the same Voyage
   RPM-pacing logic as the bulk `embed.py` script -- testing 4 questions
   in a tight loop tripped the same 3 RPM throttle from Stage 2.

**Verification results (not just "did it return an answer" -- checked
against actual source text):**

- **Cytology question:** answer named ZAB-10 through ZAB-15 as a set.
  Checked all 5 retrieved chunks' raw text -- all six field codes
  genuinely present, not fabricated.
- **Lab code question:** citations (p. 5 for the core definition, pp.
  119-120 for the CLIA-number detail) matched the actual retrieved chunk
  page ranges.
- **PID segment question (the strongest evidence):** answer listed PID-0,
  1, 2, 5, 7, 8, 10 as required and explicitly excluded PID-3 and PID-4.
  Checked the source: PID-3 is marked `UO` (Use Optional) and PID-4 varies
  `O`/`UO` in *this specific document* -- not `R` like the others. Generic
  HL7 2.3 convention often treats PID-3 as close to mandatory, so a model
  leaning on general training knowledge instead of the retrieved text
  might well have included it. It followed the document's specific
  requirement codes instead -- real grounding, not "sounds right in
  general."
- **France question:** correctly refused, explicitly stating the excerpts
  don't cover the topic. Notable: similarity scores for this query
  (0.16-0.18) were dramatically lower than every in-scope question
  (0.45-0.68) -- the score gap alone is a strong out-of-scope signal.
  Right now the system prompt does 100% of the refusal work; a cheap
  numeric guard ("skip the Claude call if best similarity < ~0.3") could
  catch obviously out-of-scope questions before paying for generation --
  noted as a good future refinement, ties directly to the "similarity
  threshold" concept flagged as unimplemented in Stage 4.

**The full manual pipeline works end-to-end**, PDF to grounded, cited
answer, verified against source text rather than just trusted at face
value.

---

## Stage 4 — Retrieval logic

**File:** `src/retrieve.py`

**What it does:** `retrieve(question, top_k=5)` embeds a question with
Voyage (`input_type="query"`), runs a pgvector `<=>` (cosine distance)
query against `document_chunks`, and returns the top-k chunks with a
`similarity` score (`1 - distance` -- exact, not approximate, since
pgvector defines `<=>` as literally `1 - cosine_similarity`).

**Key decisions:**

- **`input_type="query"` on the question, matching `"document"` used on
  the corpus in Stage 2.** The asymmetric-embedding payoff -- question and
  answer land in comparable vector-space regions despite different surface
  text.
- **`top_k=5` default.** Same trade-off as chunk size, one level up: too
  few risks missing the right passage, too many bloats Stage 5's context
  with irrelevant text.

**Tested against 3 questions chosen from content already inspected** (not
arbitrary questions -- a real basis to judge correctness):

1. *"What should be entered if a previous cytology result is not known?"*
   -> correct chunk (ZAB-13, a custom result-tracking field) ranked #2,
   0.5784, barely behind an unrelated field (0.5798). Root
   cause: this section is dozens of near-identical Y/N field templates
   sharing the same boilerplate sentence ("If the answer to this question
   is not known this field should be blank..."). The text that actually
   distinguishes one field from the next is a small fraction of each
   chunk, so embeddings of adjacent fields are nearly indistinguishable.
   Real, document-specific limitation: heavily templated field-definition
   tables (common in HL7/EDI specs) resist fine-grained retrieval.
2. *"What does a lab code represent?"* -> correct chunk (chunk-0007,
   containing the exact defining sentence) WAS the #1 result, but only
   scored 0.498 -- notably lower than the other two questions' best
   scores. Initially misread this as a miss because a debug print
   truncated the chunk to 300 characters, which happened to cut off
   before reaching the relevant sentence -- always check full chunk text,
   not a truncated preview. Root cause of the *low score* despite being
   correct: this chunk mixes three unrelated things (HL7's mailing
   address, a document-versioning disclaimer, and the actual lab-code
   definition) because all three fit under the 500-token budget, so the
   greedy chunker never split them. Real, concrete instance of the
   embedding-dilution concept from the earlier follow-up deep dive --
   token-count-only chunking can't see a section-header boundary
   ("3 SUPPLEMENTAL NOTES") that a human would obviously split on.
3. *"What is required in the PID segment for patient identification?"* ->
   clean win, chunk-0027 explicitly contains "6.1.4 Example Segment - Sample
   Identification Fields," highest similarity of all three tests (0.6831). No
   dilution, no repetition -- distinctive text retrieves well.

**Decision:** retrieval works well enough to build the full pipeline
end-to-end (all 3 questions found the right chunk in the top 3). Chose to
log both root causes as known limitations / backlog rather than pause to
rebuild the chunker -- worth revisiting once the complete system is
working, with two concrete fix ideas on record: (a) split chunks on
section-header patterns, not just paragraph breaks and token count, to
prevent dilution; (b) some form of boilerplate deduplication/downweighting
for heavily templated sections, to fix the field-discrimination problem.

---

## Exploratory — visualizing the embeddings (not part of the pipeline)

**File:** `src/visualize_embeddings.py` -> `data/embeddings_plot.png`

Asked whether it's important to graph the vectors to see similarities.
Short answer: not for the pipeline itself -- retrieval computes cosine
similarity directly on the full 1024-dim vectors, never on a 2D picture.
But it's a genuinely useful debugging/intuition tool, with a real caveat.

**What was built:** projected all 194 embeddings from 1024 dimensions down
to 2 using both PCA and t-SNE side by side, colored by source page number,
specifically to make the caveat visible rather than just stating it.

- **PCA** is a linear projection onto the 2 directions of greatest
  variance -- "honest" in that 2D distances are a real (if incomplete)
  reflection of true 1024-dim distances. Result: a loose but real
  gradient by page position, plus one arm of mid-document (~pages 50-80)
  chunks reaching out separately.
- **t-SNE** is nonlinear, optimized to preserve *local* neighborhoods at
  the cost of global distance. Result: visually striking, tight, separated
  clusters -- including one clearly isolated group (~pages 130-140, likely
  a distinct section like an appendix). The individual clusters are real
  (those points are locally similar), but the *gaps between* clusters are
  not reliable information -- t-SNE will produce confident-looking
  separation whether or not it's meaningful in the real vector space.

**Takeaway kept for the record:** both projections agree there's real
topical structure correlated with document position, which is a good sign
the embeddings capture meaning rather than noise. But only PCA's notion of
"how far apart" bears any real relationship to the space retrieval
actually operates in -- t-SNE's apparent cluster separation should never
be used to argue two topics are "very different" or "very similar."

**Key terms:**
- **Dimensionality reduction:** any technique for projecting high-
  dimensional data (1024-dim embeddings) into fewer dimensions (2, for
  plotting) while trying to preserve some notion of structure.
- **PCA (Principal Component Analysis):** a linear dimensionality
  reduction method that finds the directions of greatest variance in the
  data. Preserves global distance relationships reasonably well; less
  visually dramatic than nonlinear methods.
- **t-SNE (t-distributed Stochastic Neighbor Embedding):** a nonlinear
  dimensionality reduction method that preserves local neighborhood
  structure at the expense of global distances. Produces visually
  striking clusters that can be misleading if over-interpreted.
- **Perplexity (t-SNE parameter):** roughly controls how many neighbors
  each point considers when t-SNE decides what's "local" -- effectively a
  neighborhood-size knob.

---

## Stage 3 — Vector storage (Supabase + pgvector)

**Files:** `src/load_vectors.py` (the pipeline code); schema/index created
via migrations run through the Supabase MCP tool, not hand-typed SQL in a
client, since project/schema provisioning isn't really "the pipeline,"
it's one-time infrastructure setup.

**What it does:** created a free-tier Supabase Postgres project
(`demo-rag`), enabled the `vector`
extension, created a `document_chunks` table with a `vector(1024)` column
matching Voyage's output dimension, then `load_vectors.py` upserts all 194
embedded chunks from `data/chunks_embedded.jsonl` into it. Finished by
adding an HNSW index for cosine distance.

**Key decisions and why:**

- **Native `vector` column type, not JSON/array.** pgvector adds
  similarity *operators* (`<->` euclidean, `<#>` negative inner product,
  `<=>` cosine distance) and index types (HNSW, IVFFlat) that only work
  against the vector type. A JSON column would force a full-table scan
  with the similarity math done in application code for every query --
  exactly what pgvector exists to avoid.
- **`ON CONFLICT (chunk_id) DO UPDATE` (upsert), not plain INSERT.** Makes
  the load script safe to re-run -- if a previous run partially failed
  (or the source chunks change later), re-running just updates existing
  rows instead of erroring on the unique constraint or creating
  duplicates. Idempotency matters for anything that might get re-run,
  which in practice is almost everything.
- **HNSW index with `vector_cosine_ops`, added *after* the data was
  loaded and verified.** Two things worth remembering: (1) the index's
  operator class has to match the distance operator used at query time
  (`<=>` for cosine) or Postgres silently won't use the index; (2) at only
  194 rows this index makes no measurable difference -- a brute-force scan
  of 194 vectors is milliseconds. It's here for the pattern (and because
  it's cheap and this table will only grow), not because it's needed yet.
  IVFFlat was the other option; it requires the table to already have
  representative data before building (it clusters based on existing
  vectors), whereas HNSW has no such ordering requirement -- one reason
  HNSW is generally the current default recommendation.
- **Schema/extension/index setup done via Supabase's management API, data
  loading done via a real psycopg2 script the user runs.** These are
  different categories of work: provisioning a database is infrastructure
  setup (analogous to clicking through a dashboard), while loading and
  querying vectors is the actual pipeline logic worth writing and running
  by hand to understand.

**Verified independently** (not just trusting the load script's own
printout) via a separate SQL query: 194 rows, 194 distinct `chunk_id`s (no
duplicates from the upsert), all 1024 dimensions, page range 1-140 (full
document).

**Another real infrastructure snag:** creating a Supabase project through
the management API skips the one-time "here's your password" screen the
web dashboard shows at creation, so the DB password had to be reset
manually via Project Settings -> Database before a direct psycopg2
connection could be made.

---

## Stage 2 — Embeddings (Voyage AI)

**File:** `src/embed.py`

**What it does:** embeds all 194 chunks with Voyage's `voyage-4` model
(1024-dimensional vectors, `input_type="document"`), writes
`data/chunks_embedded.jsonl` (each chunk's original fields plus its
`embedding` and `embedding_model`), and runs a cosine-similarity sanity
check between an adjacent same-topic chunk pair and an unrelated pair.

**Key decisions and why:**

- **`voyage-4`, general-purpose model.** Current Voyage model as of this
  build (superseded voyage-3.5); 1024 dims, 32K token context, first 200M
  tokens free -- this ~87K-token corpus costs nothing. No domain-specific
  Voyage model exists for healthcare/EDI, so general-purpose is correct.
- **`input_type="document"` for the corpus, will be `"query"` for the
  question in Stage 4.** Not cosmetic -- Voyage prepends a different
  internal instruction for each side (asymmetric embedding), trained so a
  question and its answer land in matching regions of vector space despite
  looking different as raw text. Getting this backwards doesn't error, it
  just quietly degrades retrieval.
- **Small batches (15) + explicit request pacing, not just retries.**
  Learned the hard way: a new Voyage account with no payment method is
  throttled to 3 requests/min and 10K tokens/min. Proactively spacing
  requests ~21s apart is both more polite to the API and faster overall
  than firing fast and recovering from 429s.

**Sanity check result:** adjacent same-topic chunks (0080/0081) scored
0.8598 cosine similarity; unrelated chunks (front matter vs. field codes,
0002/0080) scored 0.6146. The *relative* gap is what matters, not the
absolute numbers -- text embeddings from the same document share a
vocabulary/style baseline, so even "unrelated" pairs sit fairly high.
Retrieval quality depends on relevant chunks consistently outranking
irrelevant ones for a given query, which this gap demonstrates.

**Real infrastructure detour, worth remembering:** both my execution
environments (this sandbox and the cloud container) turned out to be
behind an org-wide network egress allowlist that permits pypi.org and
api.anthropic.com but blocks third-party APIs like api.voyageai.com and
api.supabase.com entirely (confirmed 403/timeout from both). No admin
override available on a personal plan. Had to run `embed.py` from a normal
Mac Terminal instead (same project folder, not a copy -- Cowork mounts it
live). That surfaced three more real issues in sequence:
1. `psycopg2-binary` tried to build from source and failed on missing
   `pg_config` -- caused by an outdated system `pip` (19.2.3) not
   resolving the right prebuilt wheel. Fixed with `pip install --upgrade
   pip`.
2. `voyageai>=0.3.0` had no compatible release for the system's Python
   3.8 -- newer releases dropped 3.8 support. Loosening the pin didn't
   fully fix it either:
3. Even an "installable" older `voyageai` version crashed on import with
   `TypeError: 'ABCMeta' object is not subscriptable` -- the library's code
   uses `collections.abc.Awaitable[...]` generic subscripting, a Python
   3.9+ feature (PEP 585), regardless of what the package metadata claimed
   was compatible. Real fix: install Python 3.11 via Homebrew and rebuild
   the venv on it, not chase older package pins further.

Lesson for the "why pgvector/hosted Postgres" story in an interview: real
projects routinely run across multiple network zones with different
egress rules (exactly the kind of thing a vendor-connectivity/EDI pipeline
deals with too) -- a pipeline design has to account for *where* each step
is allowed to run, not just what the step does.

---

## Stage 1 — Document ingestion & chunking

**File:** `src/ingest.py`

**What it does:** extracts text from the PDF page-by-page (keeping page
numbers attached), cleans up PDF-extraction artifacts, and splits the text
into ~500-token overlapping chunks, each tagged with its source page range.
Output: `data/chunks.jsonl` — one JSON object per chunk.

**Key decisions and why:**

- **Page-level extraction with page numbers preserved as metadata.**
  Without this, there's no way to later cite "page 42" in a generated
  answer — provenance has to be carried from the very first step or it's
  gone for good.
- **Chunk size ~500 tokens, ~75 token (15%) overlap.** Too small and a
  chunk loses context (e.g. "see section 4.2" with no section 4.2 nearby).
  Too large and the chunk's embedding blurs together multiple unrelated
  ideas, which hurts similarity search, and wastes context-window space
  once retrieved. ~300–800 tokens is a common sweet spot for prose/spec
  documents. The overlap exists so a sentence sitting on a chunk boundary
  still appears whole in at least one chunk.
- **Recursive fallback splitting (paragraph → sentence → hard character
  cut).** My first version, splitting only on blank lines, produced chunks
  up to 1318 tokens because this spec has dense un-paragraphed blocks (HL7
  field tables) that pypdf returns as one giant "paragraph." Adding a
  recursive fallback — try paragraph breaks, then sentence breaks, then a
  hard character slice as a last resort — brought the max down to 571
  tokens. This is the same core idea behind LangChain's
  `RecursiveCharacterTextSplitter`.
- **Token counting via `len(text) // 4` instead of tiktoken.** tiktoken
  needs to download its encoding file from Azure blob storage on first use,
  and this environment's network allowlist blocked that call. Switched to
  the standard "~4 characters per token" heuristic for English text.
  Reasonable, because tiktoken's cl100k_base isn't even Voyage's or
  Claude's actual tokenizer either — it was always an approximation, just
  used for "is this chunk roughly the right size," not exact token math.

**Result:** 194 chunks from the 140-page PDF, token counts ranging
106–571 (avg ~450), each with `chunk_id`, `source_file`, `page_start`,
`page_end`, `token_count`, `text`.

---

## Follow-up deep dive — why the token heuristic isn't exact

Asked myself: why is `len(text) // 4` only an approximation? Two reasons:

1. **Tokenizers count learned vocabulary pieces, not characters.** Common
   English (e.g. "hospital") usually collapses into one token; rare strings
   get shredded into several small pieces. This document is full of the
   rare-string case: HL7/EDI field codes like `ZAB-13`, `PID-3`, `ORC-1`
   aren't real words, so a general-purpose tokenizer likely splits them
   into several short tokens each (e.g. `Z`/`CY`/`-`/`33`). PDF table
   debris (repeated spaces, `|` characters) adds similar overhead. So
   code/table-dense chunks burn tokens faster per character than prose
   does, and a single constant like `/4` can't know that.
   - Tried to prove this directly with real data: pulled a dense chunk
     (ZCY field list) and a prose chunk, and called Voyage's own
     `client.count_tokens(...)` on both. Both came back `403 Forbidden` --
     Voyage's tokenizer isn't bundled either; it downloads its vocabulary
     from the network on first use, same as tiktoken in Stage 1, and that
     host is also outside this sandbox's allowlist. Nice confirmation that
     "exact token count" is fundamentally a lookup-table operation, not a
     formula.
2. **"Exact" is only exact per-model.** Claude, GPT, and Voyage's models
   each have their own vocabulary, so the same text has a different real
   token count under each one. There's no single model-agnostic ground
   truth to approximate toward.

**When the heuristic is fine vs. not:** fine for chunk *sizing* toward a
fuzzy target range (being off 10-30% just nudges a boundary). NOT fine for
checking a hard API context-window limit before a call -- that needs the
real tokenizer for whichever model enforces that specific limit, because
being wrong there causes an actual failure.

---

## Follow-up deep dive #2 — precisely why chunk size affects retrieval quality

Sharpened this beyond "too small/too big is bad":

- **Too small:** not really about the retriever "not knowing which half" of
  a passage a question refers to. The real failure is a chunk can be an
  incomplete unit of meaning -- e.g. an answer fragment ("'Y' - Yes / 'N' -
  No") separated from the field code it describes (`ZAB-13`). Its
  embedding has nothing distinctive to grab onto, so it may not even
  surface for a relevant query; and if it does surface, the LLM gets an
  ambiguous fragment it can't fully interpret. Also multiplies total chunk
  count -> more embeddings to store/search for little benefit.
- **Too large:** the mechanism is **embedding dilution** -- one vector
  trying to represent several unrelated ideas ends up as a blurry average,
  not sharply close to any single one, so it tends to under-rank for
  specific queries rather than merely producing "multiple competing
  matches." Separately, a large retrieved chunk eats more of the LLM's
  context budget (crowding out other useful chunks) and runs into the
  **"lost in the middle"** effect -- LLMs are measurably worse at using
  information buried mid-context than info near the start/end, so a
  relevant sentence can get effectively ignored even when it's technically
  present.
- **Shared principle:** a good chunk is one coherent, complete idea --
  small enough to keep the embedding specific, large enough to keep the
  idea intact.

---

## Key terms — Stage 1

- **RAG (Retrieval-Augmented Generation):** an architecture where, instead
  of relying only on what a language model memorized during training, you
  retrieve relevant text from your own documents at query time and feed it
  to the model as context, so it can answer grounded in real source
  material.
- **Document ingestion:** the process of pulling raw content out of a
  source file (here, a PDF) and turning it into plain text the rest of the
  pipeline can work with.
- **Chunking:** splitting a long document into smaller, roughly
  self-contained pieces ("chunks") because embedding models and LLM context
  windows both work better on small, focused pieces of text than on one
  giant document.
- **Chunk overlap:** deliberately repeating a small amount of text between
  consecutive chunks so an idea or sentence that falls near a chunk
  boundary still shows up intact in at least one chunk.
- **Token:** the unit a language model actually processes — roughly a word
  or word-piece, not a character. Model pricing, context limits, and chunk
  sizing are usually expressed in tokens, not characters or words.
- **Tokenizer:** the specific algorithm that converts text into tokens.
  Different models (Claude, GPT, Voyage's embedding models) use different
  tokenizers, so token counts for the "same" text can differ slightly
  between them.
- **Context window:** the maximum number of tokens a model can take in as
  input (plus, for generation, produce as output) in one call. It's the
  hard ceiling that chunk size and retrieval count have to respect.
- **Metadata / provenance:** data *about* a chunk (which file, which page
  range, an ID) carried alongside the chunk text itself, so you can trace
  any retrieved passage back to its source — this is what makes citations
  possible later.
- **Recursive character/text splitting:** a chunking strategy that tries
  the most "natural" split point first (e.g. paragraph breaks) and only
  falls back to a cruder split (sentences, then raw character cuts) when
  the natural one produces a piece that's still too big.
- **Heuristic:** an approximate, "good enough" rule used in place of an
  exact calculation when the exact one is impractical or unnecessary —
  e.g. estimating tokens as `characters / 4` instead of running a real
  tokenizer.
- **Embedding** *(previewed here, covered fully in Stage 2)*: a numeric
  vector representation of a piece of text such that texts with similar
  meaning end up as vectors that are close together in that vector space.
- **Semantic search / cosine similarity** *(previewed here, covered fully
  in Stage 4)*: finding relevant text by comparing embedding vectors
  (typically with cosine similarity) rather than matching exact keywords.

---

## Key terms — Stage 2

- **Embedding (full definition):** a fixed-length vector of floats (1024
  numbers here) produced by a neural network trained so that texts with
  similar meaning produce vectors that are numerically close together in
  that vector space -- regardless of shared vocabulary. This is what
  enables semantic search instead of keyword matching.
- **Embedding dimension / vector space:** the length of the embedding
  vector (1024 for voyage-4) -- effectively the number of "axes" the model
  uses to represent meaning. Higher dimension can capture more nuance but
  costs more to store/compare.
- **Cosine similarity:** a measure of how similar two vectors' *directions*
  are (ignoring magnitude), ranging -1 to 1, computed as the dot product
  divided by the product of the vectors' lengths. The standard way to
  compare embeddings. Absolute values are domain-dependent (see the Stage 2
  sanity check note) -- relative ordering across candidates is what matters
  for retrieval.
- **Asymmetric embedding / `input_type`:** embedding a search query and the
  documents it should match using different internal instructions (Voyage:
  `input_type="query"` vs `"document"`), because a question and its answer
  often don't resemble each other as raw text even though they're
  semantically linked.
- **Rate limiting (RPM/TPM):** API providers cap how many requests-per-
  minute and tokens-per-minute an account can send, often lower for new/
  unverified accounts. Production code has to plan around this (batching,
  pacing, backoff), not just assume every call succeeds instantly.
- **Batching:** grouping multiple items into one API call instead of one
  call per item -- reduces overhead and (usually) cost, but has to respect
  the provider's own batch-size and token limits.
- **Exponential backoff:** a retry strategy where the wait time between
  retries grows (e.g. doubles) after each failure, used so retries don't
  hammer an already-struggling or rate-limited service.

---

## Key terms — Stage 3

- **pgvector:** a Postgres extension adding a native `vector` type,
  similarity operators, and index types for nearest-neighbor search --
  turns plain Postgres into a vector database.
- **Vector database / vector store:** a database (or extension of one)
  purpose-built to store embeddings and efficiently answer "which stored
  vectors are closest to this query vector," instead of doing that math in
  application code against every row.
- **Distance/similarity operators (`<->`, `<#>`, `<=>`):** pgvector's three
  comparisons -- Euclidean distance, negative inner product, and cosine
  distance respectively. Used `<=>` throughout, matching Stage 2's cosine
  similarity.
- **ANN (Approximate Nearest Neighbor) search:** exact nearest-neighbor
  search means comparing a query against every row -- expensive at scale.
  ANN indexes trade a small accuracy loss for large speed gains by only
  checking a well-chosen subset of candidates.
- **HNSW (Hierarchical Navigable Small World):** the ANN index type used
  here -- graph-based, no requirement for representative data before
  building, generally today's default recommendation.
- **IVFFlat:** pgvector's other ANN index type -- clusters vectors into
  buckets and searches only the nearest ones; requires the table to
  already hold representative data before building.
- **Operator class:** tells an index which operator/distance function it
  accelerates (`vector_cosine_ops` for `<=>` here). Index and query must
  agree on this or Postgres silently skips the index.
- **Upsert (`INSERT ... ON CONFLICT DO UPDATE`):** insert a row, or update
  it in place if a row with the same unique key already exists -- makes a
  load script safe to re-run.
- **Idempotency:** the property that running an operation multiple times
  has the same effect as running it once -- important for anything that
  might be retried after a partial failure.
- **Migration (schema migration):** a named, versioned, explicitly-applied
  change to a database's schema, as opposed to an ad hoc unrecorded one.
- **Managed/hosted Postgres:** a Postgres database provisioned and run by
  a provider (Supabase) rather than self-hosted.
- **Connection string:** the URL-shaped string
  (`postgresql://user:password@host:port/dbname`) telling a client how to
  connect to a specific database.

---

## Key terms — Stage 4

- **Retrieval (applied):** the process of embedding a query and searching
  stored document embeddings to return the most relevant chunks -- what
  `retrieve()` does.
- **Top-k retrieval:** returning a fixed number (`k`) of the
  highest-scoring results rather than a variable number based on a quality
  cutoff. Simple, but always returns `k` results even if none are a good
  match.
- **Recall (IR sense):** whether the relevant item was found among the
  results at all. All 3 test questions passed this check -- correct chunk
  landed in the top 3 every time.
- **Precision (IR sense):** of the results actually returned, how many are
  genuinely relevant. Diverges from recall -- Q1 (cytology) passed recall
  but showed weak precision, since 2 of 3 results were tangential
  boilerplate fields, not the specific answer.
- **Boilerplate/template collision:** a retrieval failure mode found in
  testing -- when many chunks share so much identical repeated text
  (common in heavily templated documents), their embeddings become nearly
  indistinguishable, making fine discrimination hard regardless of
  embedding quality.
- **Semantic/structure-aware chunking:** proposed fix for the dilution
  problem -- chunking that respects section-header/topic boundaries in
  addition to token count, instead of only checking size.
- **Similarity threshold:** a cutoff below which a retrieved result is
  treated as "not actually relevant" and discarded or triggers a fallback,
  rather than always returning top-k regardless of match quality. Not yet
  implemented -- directly relevant to Stage 5's refusal-behavior testing.

---

## Key terms — Stage 5

- **System prompt:** a separate instruction channel (distinct from the
  user message) setting persistent behavioral rules for the model -- here,
  the grounding rules. What turns a normal chatbot call into a
  RAG-constrained one.
- **Grounding:** constraining a model's answer to be derived from
  specific supplied source material rather than general training
  knowledge -- the entire purpose of RAG. The PID segment test (correctly
  excluding PID-3/PID-4 despite generic HL7 convention) was the clearest
  proof this was actually happening.
- **Hallucination:** a model generating plausible-sounding content not
  actually supported by (or contradicting) the source material -- the
  specific failure mode grounding instructions exist to prevent.
- **Refusal / abstention:** the model explicitly declining to answer
  rather than guessing, when retrieved context doesn't support an answer.
  Tested directly with the France question.
- **Citation / traceability:** tracing a generated claim back to its
  specific source location (page numbers here) -- the payoff of carrying
  page metadata all the way from Stage 1 through to the final answer.
- **Identity-linked API key vs. workspace-scoped key:** a key tied to a
  person's account (works across their workspaces, needs a per-request
  workspace header unless scoped at creation) versus a key belonging to
  one workspace outright, needing no such header.
- **Reasoning effort (`output_config.effort`):** a model-dependent
  parameter controlling computation/reasoning depth -- the current API's
  replacement concept for `temperature`, though a genuinely different axis
  (depth vs. randomness), not just a rename.

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

## System design notes -- beyond this repo (interview prep)

Not implemented here, but worth writing down while it's fresh -- a set of
questions that came up thinking about how this project would actually be
discussed in an interview, and how it'd need to change to go from "one
document, one pipeline" to something closer to a real product.

**Quick vocabulary fix:** CLI = command-line interface, not "client
language interface." A CLI wrapper for `generate.py` would be worth
building purely to remove friction for anyone poking at the repo (right
now you have to hand-edit `test_questions` or import the functions
yourself to ask a custom question) -- but it's polish, not a skill
demonstration. The things in this repo that actually say something about
RAG understanding are the chunk-dilution finding + header-flush fix, and
the LangChain rebuild's source-verified findings (the `embed_documents()`
/ `embed_query()` tokenize-call-storm and rate-limit gaps, and
`ChatAnthropic` quietly avoiding the `temperature` API error the manual
pipeline had to debug by hand). Those are the interview story; the CLI
isn't.

**Comparing the manual pipeline to the LangChain rebuild, properly:**
what's here (reading the four answers side by side, reasoning from
library source about *why* they differ) is honest but qualitative. A
step up, without much extra work, would be a small eval harness: hand-pick
10-15 questions against content I've actually verified in the document
(same principle as the existing 3-4 test questions, just more of them),
label which chunk(s) should come back for each, then measure:
  - **Retrieval quality:** precision@k / recall@k / hit-rate@k (did the
    right chunk show up in the top k), MRR or nDCG if ranking order
    matters, not just presence.
  - **Faithfulness / groundedness:** does the generated answer stay
    inside the retrieved context, or drift into the model's general
    knowledge. RAGAS is the standard open-source framework for this kind
    of LLM-as-judge scoring (faithfulness, answer relevancy, context
    precision, context recall are its named metrics); at this project's
    scale, a consistent manual rubric would do the same job.
  - **Operational numbers I already have "for free":** 226 vs. 216
    chunks, and the real `RateLimitError` hit and fixed in both
    pipelines at the same point in the code -- worth stating explicitly
    as evidence, not just narrative color.

**Chunking strategy -- what's here vs. what else exists:** this repo uses
fixed token-size windows with overlap, plus a hand-rolled header-flush
heuristic (src/) or `RecursiveCharacterTextSplitter`'s built-in separator
cascade (src_langchain/). Other approaches, and where they'd actually
beat this one:
  - **Semantic chunking** -- split where consecutive sentences'
    embeddings diverge, instead of at a fixed size. Costs more compute
    (an embedding call per boundary decision) but adapts to uneven
    paragraph density instead of assuming ~500 tokens is always the
    right unit.
  - **Parent-document / small-to-big retrieval** -- embed and search
    over small, precise chunks, but hand the LLM the larger surrounding
    section for generation. Solves precision and context-richness at
    once; the tradeoff is a slightly more complex retrieval step
    (fetch small chunk, then look up its parent).
  - **Structure-aware chunking** -- split on the document's own
    structure (numbered sections, in this spec's case) instead of raw
    character/token counts. Would sidestep the whole header-flush
    problem by construction rather than patching it after the fact.
  - **Table-aware chunking** -- matters specifically for a technical
    spec like this one, which has field-definition tables. Naive text
    chunking can mangle tabular data into unreadable fragments;
    specialized extraction (e.g. converting tables to Markdown before
    chunking) is a common fix I didn't need here but would for a
    table-heavy document.
  Which one's "right" depends on document type: legal contracts want
  clause-level boundaries (cross-references matter), codebases want
  function/class boundaries, scanned documents need OCR-aware layout
  parsing before chunking is even meaningful.

**Ingesting a whole folder of documents, not just one PDF:** loop over
files instead of hardcoding `SOURCE_PDF_PATH`, and the one thing that
actually matters is tagging every chunk's metadata with which source
file it came from, so citations point to "page 4 of contract_A.pdf," not
just "page 4." Two things worth building in at that scale that don't
matter at "one document": prefixing `chunk_id` with the filename (or a
hash) to avoid collisions across files, and hashing each file's contents
so an ingestion re-run skips files that haven't changed instead of
re-embedding (and re-paying for) the whole folder every time.
LangChain's `DirectoryLoader` handles "loop over a folder, pick the
right loader per file extension" out of the box if leaning on the
framework there.

**Multi-tenant scaling (e.g. a law firm with many clients, each needing
separate instances):** one generic, parameterized pipeline, not a
separate copy per client -- isolation happens at the data layer, not the
code layer. Standard multi-tenant SaaS patterns: the **pool model** (one
shared table, every row tagged `tenant_id`, every query filtered by it),
the **silo model** (a fully separate table/schema/database per tenant),
and hybrids of the two. This repo already demonstrates a small-scale
silo pattern -- the manual pipeline and the LangChain rebuild write into
two separate Postgres collections specifically so neither can touch the
other's data. Generalizing that same idea to "one collection per client"
is a direct extension of something already built here, not a new
concept. For something like a law firm specifically, the pool model's
risk is real: one missed `WHERE tenant_id = X` clause and one client's
privileged documents leak into another client's answer -- not just a
bug, a conflict-of-interest/malpractice-adjacent event. That's a case
where the more expensive, fully-isolated silo architecture is usually
the right call over the cheaper shared one, despite higher operational
cost.

---

## Environment notes

- Dev sandbox has no `sudo`/Docker access, so a local Postgres server isn't
  possible here — hence the hosted Supabase project for Stage 3.
- Outbound network is allowlisted; some hosts (e.g. Azure blob storage,
  which tiktoken uses) are blocked. PyPI is reachable, so `pip install`
  works normally.
