# Hybrid RAG with pgvector + Cohere Rerank-3.5

> **A retrieval-augmented Q&A system over technical documentation. Combines pgvector semantic search, Postgres full-text search, and a Cohere cross-encoder reranker, then answers with citation-grounded GPT-4o-mini.**

Built with hybrid retrieval (dense + sparse fused via Reciprocal Rank Fusion), an optional rerank stage that suppresses cross-source keyword collisions, inline citation grounding traceable back to source chunks, and end-to-end LLM observability via Langfuse.


## What it does

LLMs are great at synthesizing answers but miss specifics in fast-moving SDK docs — they invent APIs that don't exist or describe last year's behavior. HybridRAG indexes a corpus of technical docs and answers natural-language questions with inline `[chunk_<id>]` citations back to the exact source.

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

A constant `k=60` flattens the contribution of any single very-high rank, so the fusion stays stable even when one signal disagrees sharply.


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

Pure vector search misses queries that hinge on specific API names — embeddings smooth over the exact tokens you care about. Pure keyword search misses paraphrases. RRF fusion takes the top-50 from each, sums `1 / (60 + rank)` across both rankings, and returns whichever chunks accumulate the most score — no learned weights, no tuning.

### Cohere rerank-3.5 as the final filter

RRF's failure mode is that a strong sparse hit in the *wrong source* can pollute the fused top — e.g., a "What is a Pydantic BaseModel?" query where FastAPI's body tutorial mentions "BaseModel" and outranks the canonical Pydantic doc. Cohere's `rerank-v3.5` is a cross-encoder: it reads the query and each candidate's full text together and produces a relevance score that catches semantic mismatches RRF's bag-of-rankings approach can't. Used as a final pass over the top-20 hybrid candidates to produce the final top-5.

### Citation grounding

The system prompt requires every factual claim be tagged with `[chunk_<id>]`. After generation, those markers are regex-extracted and resolved back to chunk metadata, so the API response includes a `sources` list of only the chunks the answer actually used — not just everything retrieved.

### Observability via Langfuse

Every `/ask` request is one Langfuse trace. The `@observe()` decorator wraps the endpoint and each retriever as named spans; the `langfuse.openai` drop-in client auto-logs every embedding and chat-completion call with token counts, cost, and latency. Result: a tree showing exactly which step was slow or which chunks were fed to the LLM, viewable at cloud.langfuse.com. Disabled cleanly if keys aren't set.

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

Markdown only, scraped once via `git clone` + filename flattening (`docs/tutorial/cors.md` → `tutorial__cors.md`). Not committed to the repo (gitignored) — re-fetched on demand.
