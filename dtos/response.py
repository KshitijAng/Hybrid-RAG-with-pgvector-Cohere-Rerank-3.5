"""Outgoing API shapes — the DTOs the caller receives back from POST /ask."""

from pydantic import BaseModel, Field


class Source(BaseModel):  # one cited chunk in the answer's `sources` list
    id: int = Field(..., description="Chunks table primary key — matches a `[chunk_<id>]` marker in the answer.")
    source: str = Field(..., description="Which doc set the chunk came from (e.g., 'fastapi', 'langchain').")
    doc_id: str = Field(..., description="The flattened source filename (e.g., 'tutorial__cors.md').")
    chunk_index: int = Field(..., description="Position within the source file (0, 1, 2, ...).")
    title: str | None = Field(default=None, description="Nearest heading above the chunk; may be null.")


class AskResponse(BaseModel):  # full /ask payload
    answer: str = Field(
        ...,
        description="LLM-generated answer grounded in retrieved chunks, with inline [chunk_<id>] citations.",
    )
    sources: list[Source] = Field(
        ...,
        description="Only the chunks the answer actually cited (resolved from [chunk_<id>] markers).",
    )
    retrieval_mode: str = Field(
        ...,
        description="Which retrieval strategy was used: dense / sparse / hybrid.",
    )
    n_chunks: int = Field(
        ...,
        description="How many chunks were retrieved and shown to the LLM (≤ requested k).",
    )
    latency_ms: int = Field(
        ...,
        description="End-to-end latency in milliseconds — embed + retrieve + LLM + serialize.",
    )
