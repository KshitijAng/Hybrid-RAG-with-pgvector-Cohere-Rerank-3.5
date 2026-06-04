"""Retrieval functions over the chunks table.

Four modes, each returning the top-K most relevant chunks for a query:

  dense(query)          → pgvector cosine similarity (semantic match)
  sparse(query)         → Postgres FTS via ts_rank_cd  (keyword match)
  hybrid(query)         → both above fused via Reciprocal Rank Fusion (RRF)
  hybrid_rerank(query)  → hybrid candidates re-scored by Cohere rerank-3.5

All share the same return shape: a list of dicts with chunk metadata plus a
`score` field (interpretation differs per mode — see each docstring).
"""

import os

import cohere
from dotenv import load_dotenv
from langfuse import observe

from db import connect
from embed import embed_texts


load_dotenv()


# RRF "k" constant — typical value from the literature, keeps the math stable.
RRF_K = 60


# Cohere client is lazy so importing this module without a key still works.
_cohere_client: cohere.ClientV2 | None = None


def _get_cohere_client() -> cohere.ClientV2:
    global _cohere_client
    if _cohere_client is None:
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY is missing. Get a free key at https://cohere.com and set it in .env"
            )
        _cohere_client = cohere.ClientV2(api_key=api_key)
    return _cohere_client


def _rows_as_dicts(cursor) -> list[dict]:
    """Turn psycopg's row tuples into list[dict] using cursor.description."""
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


@observe()
def dense(query: str, k: int = 50) -> list[dict]:
    """Pure vector search. `score` is cosine distance (lower = more similar)."""
    [query_vec] = embed_texts([query])
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source, doc_id, chunk_index, title, text,
                   embedding <=> %s::vector AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vec, query_vec, k),
        )
        return _rows_as_dicts(cur)


@observe()
def sparse(query: str, k: int = 50) -> list[dict]:
    """Full-text keyword search. `score` is ts_rank_cd (higher = more relevant)."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source, doc_id, chunk_index, title, text,
                   ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score
            FROM chunks
            WHERE tsv @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, query, k),
        )
        return _rows_as_dicts(cur)


@observe()
def hybrid(query: str, k: int = 20, candidates_per_mode: int = 50) -> list[dict]:
    """RRF fusion of dense + sparse rankings. `score` is the fused RRF score."""
    [query_vec] = embed_texts([query])

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            WITH dense AS (
                SELECT id,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            ),
            sparse AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC
                       ) AS rank
                FROM chunks
                WHERE tsv @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC
                LIMIT %s
            ),
            fused AS (
                -- FULL OUTER JOIN keeps every chunk that appears in EITHER ranking.
                SELECT
                    COALESCE(d.id, s.id) AS id,
                    COALESCE(1.0 / ({RRF_K} + d.rank), 0)
                  + COALESCE(1.0 / ({RRF_K} + s.rank), 0) AS score
                FROM dense d
                FULL OUTER JOIN sparse s USING (id)
            )
            SELECT c.id, c.source, c.doc_id, c.chunk_index, c.title, c.text, f.score
            FROM fused f
            JOIN chunks c USING (id)
            ORDER BY f.score DESC
            LIMIT %s
            """,
            (
                query_vec, query_vec, candidates_per_mode,
                query, query, query, candidates_per_mode,
                k,
            ),
        )
        return _rows_as_dicts(cur)


@observe()
def hybrid_rerank(query: str, k: int = 5, candidates: int = 20) -> list[dict]:
    """Hybrid retrieval + Cohere rerank-3.5.

    1. hybrid(query) returns `candidates` chunks via RRF fusion.
    2. Cohere's cross-encoder re-scores them against the query.
    3. Return the top `k` sorted by Cohere relevance.

    `score` here is Cohere's relevance score in [0, 1].
    """
    pool = hybrid(query, k=candidates)
    if not pool:
        return []

    client = _get_cohere_client()
    response = client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=[c["text"] for c in pool],
        top_n=k,
    )

    reranked: list[dict] = []
    for result in response.results:
        chunk = dict(pool[result.index])
        chunk["score"] = result.relevance_score
        reranked.append(chunk)
    return reranked
