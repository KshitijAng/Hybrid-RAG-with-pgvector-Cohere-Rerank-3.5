"""Incoming request DTOs for the HybridRAG API.

Currently one shape: a natural-language query that the user wants answered
against the indexed corpus. The caller can pick a retrieval mode and how
many chunks to feed the LLM.
"""

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):  # body of POST /ask
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The user's natural-language question.",
    )
    mode: Literal["dense", "sparse", "hybrid", "hybrid_rerank"] = Field(
        default="hybrid_rerank",
        description="Retrieval strategy — dense / sparse / hybrid (RRF) / hybrid_rerank (RRF + Cohere rerank).",
    )
    k: int = Field(   # how many chunks to feed the LLM
        default=5,
        ge=1,
        le=20,
        description="How many top chunks to retrieve and pass to the LLM.",
    )
