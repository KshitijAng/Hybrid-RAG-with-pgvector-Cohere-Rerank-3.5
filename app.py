"""HybridRAG HTTP API.

POST /ask    — retrieve relevant chunks for a query and return a grounded LLM answer with citations.
GET  /health — liveness probe, doesn't touch DB or LLM.

Run:
    venv/bin/uvicorn app:app --reload --port 8000

# Langfuse is used for tracing, monitoring, and evaluating LLM workflows.
# Traces every /ask request as a tree:
#   ask → retrieval → embed/SQL → LLM → response

# Wiring:
#   - @observe() on this endpoint + retrievers + answer()  → named spans
#   - langfuse.openai.OpenAI drop-in in embed.py + llm.py   → auto-logs each API call
# Traces ship in the background, never block the request path.
"""

import time

from fastapi import FastAPI, HTTPException
from langfuse import observe

from dtos.request import AskRequest
from dtos.response import AskResponse
from llm import answer
from retrieve import dense, hybrid, hybrid_rerank, sparse


app = FastAPI(
    title="HybridRAG",
    description="Hybrid-search RAG over technical docs (LangChain, FastAPI, Pydantic, HuggingFace, etc).",
    version="1.0.0",
)


# Mode string -> retrieval function (so we can pick by name from the request body).
_RETRIEVERS = {
    "dense":         dense,
    "sparse":        sparse,
    "hybrid":        hybrid,
    "hybrid_rerank": hybrid_rerank,
}


@app.get("/health")
def health() -> dict:
    """Cheap liveness probe — no DB or LLM call."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@observe()
def ask(req: AskRequest) -> AskResponse:
    """Retrieve top-K chunks for `query`, then ask the LLM to answer using them."""
    t0 = time.time()

    retriever = _RETRIEVERS[req.mode]
    chunks = retriever(req.query, k=req.k)

    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks matched this query.")

    result = answer(req.query, chunks)

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieval_mode=req.mode,
        n_chunks=len(chunks),
        latency_ms=int((time.time() - t0) * 1000),
    )
