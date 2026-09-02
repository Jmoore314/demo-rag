"""
Stage 5: Prompt construction and grounded generation with Claude.

Takes the chunks Stage 4 retrieved for a question, builds a prompt that
forces the model to answer only from those chunks, and calls Claude to
produce the final answer. This is the step that turns "here are 5 relevant
passages" into "here is an answer to your actual question" -- and the step
where RAG can quietly fail back into being a regular chatbot if the prompt
doesn't explicitly force grounding.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

import anthropic

from retrieve import retrieve


def _require_env(*names: str) -> None:
    """Fail with a plain, actionable message instead of a bare KeyError --
    this repo is meant to be read and adapted, not cloned-and-run without
    your own document and API keys (see README.md's Setup section)."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"Missing required .env value(s): {', '.join(missing)}")
        print("See README.md's Setup section -- this pipeline needs your own API keys.")
        sys.exit(1)

# Current as of this build -- check platform.claude.com/docs/en/about-claude/models/overview
# for the latest recommended model id, these change over time.
MODEL = "claude-sonnet-4-5"

# This system prompt is the entire mechanism that makes this "RAG" instead
# of "a chatbot that happens to have some text pasted above the question."
# Without an explicit instruction to refuse ungrounded answers, a capable
# model will happily blend the provided context with its own training-data
# knowledge of HL7/EDI in general -- which defeats the purpose, since we
# specifically want answers traceable to *this* document, not "probably
# true in general."
SYSTEM_PROMPT = """You are answering questions about a technical HL7/EDI \
specification document using only the excerpts provided below.

Rules:
- Answer ONLY using information present in the provided excerpts. Do not \
use outside knowledge about HL7, EDI, or lab systems in general, even if \
you believe it's correct -- the goal is an answer grounded in this \
specific document, not general knowledge.
- If the excerpts don't contain enough information to answer the \
question, say so explicitly rather than guessing or filling gaps.
- Cite the page number(s) for every claim, using the format (p. X) or \
(pp. X-Y), so the answer can be checked against the source.
- Be concise and direct."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Formats retrieved chunks as labeled, citable excerpts. Each excerpt
    is tagged with its page range up front, in the text itself -- not just
    in metadata Claude never sees -- because the model can only cite pages
    it's actually been shown."""
    context_blocks = []
    for c in chunks:
        pages = (
            f"p. {c['page_start']}"
            if c["page_start"] == c["page_end"]
            else f"pp. {c['page_start']}-{c['page_end']}"
        )
        context_blocks.append(f"[Excerpt from {pages}]\n{c['content']}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""Context excerpts from the document:

{context}

---

Question: {question}"""


def generate_answer(question: str, top_k: int = 5) -> dict:
    chunks = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, chunks)

    # Newer "identity-linked" API keys (tied to a person rather than a
    # specific workspace) require every request to state which workspace
    # it acts in via this header -- workspace-scoped keys don't need it,
    # so this is opt-in based on whether ANTHROPIC_WORKSPACE_ID is set.
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    default_headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=default_headers)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        # As of the current Messages API, `temperature` isn't a request
        # parameter anymore -- replaced by output_config.effort, which
        # controls reasoning *depth*, not randomness. There's no sampling
        # knob to force determinism with in this API version. `effort` is
        # also only supported on some model generations (not
        # claude-sonnet-4-5), so we're leaving output_config unset
        # entirely rather than chase per-model compatibility -- the
        # default behavior is fine for a short grounded lookup over 5
        # excerpts, which doesn't need deep multi-step reasoning anyway.
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "question": question,
        "answer": response.content[0].text,
        "chunks_used": chunks,
    }


if __name__ == "__main__":
    _require_env("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "SUPABASE_DB_URL")

    test_questions = [
        "What should be entered if a previous cytology result is not known?",
        "What does a lab code represent?",
        "What is required in the PID segment for patient identification?",
        # Deliberately out-of-scope -- the real test of whether grounding
        # actually works isn't "does it answer relevant questions well,"
        # it's "does it correctly refuse irrelevant ones instead of
        # answering from general knowledge."
        "What is the capital of France?",
    ]

    for q in test_questions:
        result = generate_answer(q, top_k=5)
        print(f"\n{'=' * 80}\nQ: {q}\n{'=' * 80}")
        print(result["answer"])
        print("\nSources used:")
        for c in result["chunks_used"]:
            print(f"  - {c['chunk_id']} (pp. {c['page_start']}-{c['page_end']}, similarity={c['similarity']:.3f})")
