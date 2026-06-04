# Hybrid RAG with pgvector + Cohere Rerank-3.5

> **A retrieval-augmented Q&A system over technical documentation. Combines pgvector semantic search, Postgres full-text search, and a Cohere cross-encoder reranker, then answers with citation-grounded GPT-4o-mini.**

Built with hybrid retrieval (dense + sparse fused via **Reciprocal Rank Fusion (RRF)**), an optional rerank stage reorders retrieved documents using a more accurate relevance model so the most contextually useful chunks are sent to the LLM, inline citation grounding traceable back to source chunks, and end-to-end LLM observability via Langfuse.


## What it does

This is a learning-focused implementation that combines semantic retrieval (dense) and keyword-based searc (sparse) to improve grounding over technical documentation, enabling natural-language question answering with inline chunk-level citations back to the original source content.

Four retrieval modes are exposed so you can see what each technique contributes to the final ranking.

**Indexed corpus** (5,753 chunks across 6 sources):

| Source | Files | Chunks |
|---|--:|--:|
| HuggingFace Transformers | 300 | 2,810 |
| FastAPI | 153 | 1,935 |
| Pydantic | 88 | 689 |
| LangChain | 26 | 167 |
| LangGraph | 16 | 120 |
| Anthropic cookbook | 13 | 32 |
| **Total** | **596** | **5,753** |


## Architecture

```
                ┌───────────────────────────────────────┐
                │  data/corpus/<source>/*.md            │
                │  (596 markdown files, 6 sources)      │
                └────────────────┬──────────────────────┘
                                 │ ingest.py
                                 ▼
              ┌───────────────────────────────────┐
              │  chunk.py  — header-aware split   │
              │  → 800-token chunks (100 overlap) │
              └────────────────┬──────────────────┘
                               │
                               ▼
              ┌───────────────────────────────────┐
              │  embed.py — text-embedding-3-small│
              │  (1536 dims, batches of 100)      │
              └────────────────┬──────────────────┘
                               │ upsert
                               ▼
           ┌────────────────────────────────────────────┐
           │  Postgres 16 + pgvector — chunks table     │
           │   id, source, doc_id, chunk_index, title,  │
           │   text, embedding (HNSW), tsv (GIN)        │
           └─────────────────┬──────────────────────────┘
                             │
                             │ retrieve.py
                             ▼
   ┌─────────────────────────────────────────────────────────┐
   │   dense   sparse   hybrid (RRF)   hybrid_rerank         │
   │     │        │          │            │                  │
   │     └────────┴──────────┴────────────┘                  │
   │                       │                                 │
   │          top-K chunks (default K=5)                     │
   └───────────────────────┬─────────────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────────────┐
              │  llm.py — gpt-4o-mini           │
              │  + citation grounding           │
              │    ([chunk_<id>] markers)       │
              └────────────────┬────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   ┌────────────────┐                  ┌──────────────────┐
   │  FastAPI       │                  │  Streamlit UI    │
   │  POST /ask     │                  │  (browser)       │
   └────────────────┘                  └──────────────────┘
```


## Retrieval modes

| Mode | How it works | Strengths | Weakness |
|---|---|---|---|
| `dense` | pgvector cosine on text-embedding-3-small | semantic match, synonym handling | misses exact API names / unusual tokens |
| `sparse` | Postgres `tsvector` + `ts_rank_cd` | exact keyword hits, ~15ms | no concept generalization |
| `hybrid` | RRF fusion of dense + sparse | catches both signal types | rewards cross-source keyword collisions |
| `hybrid_rerank` | hybrid candidates → Cohere rerank-3.5 cross-encoder | content-aware re-scoring suppresses noise | +1s latency, extra API call |

**RRF formula** (per chunk, summed across both rankings):

```
score(c) = Σ  1 / (k + rank_i(c))      with k = 60
```

A smoothing constant `k=60` keeps the fused ranking balanced by reducing the impact of any single top-ranked result.

## Tech stack

| Concern | Tool |
|---|---|
| Language | Python 3.14 |
| Validation / DTOs | Pydantic 2.x |
| Vector DB | Postgres 16 + pgvector (HNSW index, cosine) |
| Full-text search | Postgres `tsvector` + GIN index |
| Chunking | LangChain text splitters (Markdown headers → tiktoken-sized) |
| Embeddings | `text-embedding-3-small` via GitHub Models |
| LLM | `gpt-4o-mini` via GitHub Models |
| Reranker | Cohere `rerank-v3.5` |
| Observability | Langfuse (auto-traced OpenAI client + `@observe()` spans) |
| HTTP API | FastAPI + Uvicorn |
| Browser UI | Streamlit |
| Container | Docker Compose (Postgres only) |

### Services & ports

| Service | Port | Where it runs |
|---|---|---|
| **Postgres** | `5432` | Docker (`hybridrag_postgres`) |
| **FastAPI** | `8000` | Local process — docs at `/docs` |
| **Streamlit** | `8501` | Local process |


## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe — no DB / LLM call |
| `POST` | `/ask` | Retrieve top-K chunks for a query, return a grounded LLM answer + the chunks it cited |

`POST /ask` request body:

```json
{
  "query": "How do I add CORS in FastAPI?",
  "mode": "hybrid_rerank",
  "k": 5
}
```


## Highlights

### Hybrid retrieval, not just vectors

The system combines dense vector search with sparse keyword search to improve retrieval quality across technical documentation. Semantic retrieval captures conceptual similarity, while keyword search preserves exact API and identifier matches. Results from both retrieval methods are fused using Reciprocal Rank Fusion (RRF) before being passed downstream.

### Cohere rerank-3.5 as the final filter

A final reranking step using `Cohere rerank-v3.5` improves relevance by evaluating the query against candidate chunks directly. This helps prioritize the most contextually accurate documentation before generating the final response.

### Citation grounding

Generated responses include inline chunk-level citations `[chunk_<id>]` linked back to the original source documents. Referenced chunks are extracted post-generation and returned alongside the answer for transparent grounding and traceability.

### Observability via Langfuse

Langfuse tracing is integrated across retrieval and generation workflows to monitor latency, token usage, costs, and retrieved context. This provides end-to-end visibility into the RAG pipeline for debugging and evaluation.

### Models via GitHub Models

Both the embedding model (`text-embedding-3-small`) and the chat model (`gpt-4o-mini`) are accessed through [GitHub Models](https://github.com/marketplace/models), which exposes an OpenAI-compatible endpoint with a free per-day quota for prototyping. Auth is a fine-grained PAT with the `Models` scope — no separate billing setup needed.


## Corpus

All indexed docs come from public OSS documentation:

- `fastapi/` — Tiangolo's FastAPI docs (`tiangolo/fastapi/docs/en/docs/`)
- `pydantic/` — Pydantic v2 docs (`pydantic/pydantic/docs/`)
- `huggingface/` — Transformers docs (`huggingface/transformers/docs/source/en/`)
- `langchain/` — LangChain monorepo READMEs (`langchain-ai/langchain`)
- `langgraph/` — LangGraph monorepo READMEs (`langchain-ai/langgraph`)
- `anthropic/` — Anthropic cookbook (`anthropics/anthropic-cookbook`)
