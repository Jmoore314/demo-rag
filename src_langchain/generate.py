"""
LangChain rebuild -- Stage 5: prompt construction and grounded generation.

Direct comparison point for src/generate.py. `ChatPromptTemplate` +
`ChatAnthropic` composed with LCEL (`prompt | model | StrOutputParser()`)
replaces build_prompt() + the raw anthropic.Anthropic() call.

**A real, verified point in LangChain's favor here** (checked by reading
the installed langchain-anthropic source, not assumed): the same
`temperature` API change that broke the manual pipeline's first draft
(`temperature` was removed from the Messages API entirely, replaced by
`output_config.effort`) is handled transparently by ChatAnthropic. Its
`temperature` field defaults to None, and the wrapper strips every
None-valued field out of the actual request payload before calling the
API (`{k: v for k, v in payload.items() if v is not None}`). So simply
not setting `temperature=` here -- the natural way to use this class --
never sends it, and this code never hits the error the manual pipeline
had to debug and fix by hand. Worth contrasting directly with
load_vectors.py's finding about VoyageAIEmbeddings: that integration
makes no defensive accommodation for its own account's rate limits, while
this one actively insulates you from an API change. Same company
(LangChain), different partner integrations, genuinely different levels
of built-in resilience -- not something you'd know without checking both.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(dotenv_path=".env")

from retrieve import get_vector_store, retrieve


def _require_env(*names: str) -> None:
    """Fail with a plain, actionable message instead of a bare KeyError --
    this repo is meant to be read and adapted, not cloned-and-run without
    your own document and API keys (see README.md's Setup section)."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"Missing required .env value(s): {', '.join(missing)}")
        print("See README.md's Setup section -- this pipeline needs your own API keys.")
        sys.exit(1)


# Checked at import time, not just inside __main__ -- ChatAnthropic below
# is constructed at module scope, so a missing key would otherwise surface
# as a bare KeyError before this module even finishes importing.
_require_env("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "SUPABASE_DB_URL")

# Current as of this build -- same model as src/generate.py, so any
# difference in answers is attributable to the pipeline, not the model.
MODEL = "claude-sonnet-4-5"

# Identical grounding rules to src/generate.py's SYSTEM_PROMPT, adjusted
# only for this pipeline's citation granularity: chunks here never span
# multiple pages (see src_langchain/ingest.py's page-boundary note), so
# citations are always a single page, never a range.
SYSTEM_PROMPT = """You are answering questions about a technical HL7/EDI \
specification document using only the excerpts provided below.

Rules:
- Answer ONLY using information present in the provided excerpts. Do not \
use outside knowledge about HL7, EDI, or lab systems in general, even if \
you believe it's correct -- the goal is an answer grounded in this \
specific document, not general knowledge.
- If the excerpts don't contain enough information to answer the \
question, say so explicitly rather than guessing or filling gaps.
- Cite the page number for every claim, using the format (p. X), so the \
answer can be checked against the source.
- Be concise and direct."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Context excerpts from the document:\n\n{context}\n\n---\n\nQuestion: {question}",
        ),
    ]
)

# `anthropic_api_key` is read from ANTHROPIC_API_KEY automatically if
# omitted, same as the raw SDK -- passed explicitly here just to make the
# dependency visible. No `temperature` or `output_config` set -- see
# module docstring for why that's deliberate, not an oversight. If your
# key is identity-linked rather than workspace-scoped, ChatAnthropic also
# accepts `default_headers={"anthropic-workspace-id": ...}`, same fix as
# src/generate.py used -- not needed here since the project's key is
# already workspace-scoped.
model = ChatAnthropic(
    model=MODEL,
    max_tokens=1024,
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
)

# LCEL: the `|` composes prompt -> model -> output parser into a single
# runnable. StrOutputParser just extracts the plain text answer instead of
# the full message object -- the LangChain equivalent of
# response.content[0].text in the manual pipeline.
chain = prompt | model | StrOutputParser()


def build_context(chunks: list[dict]) -> str:
    blocks = [f"[Excerpt from p. {c['page']}]\n{c['content']}" for c in chunks]
    return "\n\n---\n\n".join(blocks)


def generate_answer(vector_store, question: str, top_k: int = 5) -> dict:
    chunks = retrieve(vector_store, question, top_k=top_k)
    context = build_context(chunks)
    answer = chain.invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "chunks_used": chunks}


if __name__ == "__main__":
    # Same 4 test questions as src/generate.py, including the deliberate
    # out-of-scope one -- the refusal behavior is the real grounding test.
    test_questions = [
        "What should be entered if a previous cytology result is not known?",
        "What does a lab code represent?",
        "What is required in the PID segment for patient identification?",
        "What is the capital of France?",
    ]

    store = get_vector_store()

    for q in test_questions:
        result = generate_answer(store, q, top_k=5)
        print(f"\n{'=' * 80}\nQ: {q}\n{'=' * 80}")
        print(result["answer"])
        print("\nSources used:")
        for c in result["chunks_used"]:
            print(f"  - {c['chunk_id']} (p. {c['page']}, similarity={c['similarity']:.3f})")
