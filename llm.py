"""LLM answer generation via GitHub Models.

Takes a user query + retrieved chunks, builds a context-augmented prompt,
calls a chat model, returns a grounded answer with source citations.

Shares the same OpenAI-compatible setup as embed.py.

Citation grounding: the LLM tags every factual claim with a [chunk_<id>]
marker so each statement traces back to a specific source chunk.
"""

import os
import re

from dotenv import load_dotenv
from langfuse import observe

# Drop-in for `openai` that auto-traces calls when Langfuse env vars are set.
from langfuse.openai import OpenAI


load_dotenv()


CHAT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1


_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)


SYSTEM_PROMPT = """You are a technical documentation assistant. Answer the user's question using ONLY the provided context chunks.

FORMATTING:
- Use plain text with clear headings where helpful.
- Use numbered or bulleted lists for steps and options.
- Include code examples when the chunks contain them.
- Keep answers concise and actionable — max 250 words excluding code.

CITATIONS:
- Cite the source of every factual claim inline as [chunk_<id>], e.g. [chunk_1234].
- Place each citation immediately after the claim it supports.
- If multiple chunks support one claim, cite all of them: [chunk_12][chunk_99].

IF NOT FOUND:
If the provided chunks don't contain enough information to answer, respond with exactly:
"I couldn't find relevant information in the docs for your question."

STRICT RULES:
- Use ONLY the provided chunks. No external knowledge, no assumptions.
- Don't reference "documents", "chunks", or "the docs" in your prose — write naturally as if the information is just true.
- Don't speculate, extrapolate, or invent information not stated in the chunks.
"""


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context string with citation IDs."""
    lines = []
    for c in chunks:
        header = f"[chunk_{c['id']}]  source: {c['source']}/{c['doc_id']}  —  {c['title'] or '(no title)'}"
        lines.append(header)
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)


@observe()
def answer(query: str, chunks: list[dict]) -> dict:
    """Generate a grounded answer for `query` using the supplied `chunks`.

    Returns {"answer": str, "sources": list[dict]} — `sources` lists only the
    chunks the answer actually cited (parsed from `[chunk_<id>]` markers).
    """
    context = _format_context(chunks)
    user_message = f"Context:\n\n{context}\n\nQuestion: {query}"

    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )
    answer_text = response.choices[0].message.content or ""

    cited_ids = {int(m) for m in re.findall(r"\[chunk_(\d+)\]", answer_text)}
    sources = [
        {
            "id":          c["id"],
            "source":      c["source"],
            "doc_id":      c["doc_id"],
            "chunk_index": c["chunk_index"],
            "title":       c["title"],
        }
        for c in chunks
        if c["id"] in cited_ids
    ]

    return {"answer": answer_text, "sources": sources}
