# System design notes -- beyond this repo

*(interview prep -- not implemented in this repo)*

---

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

