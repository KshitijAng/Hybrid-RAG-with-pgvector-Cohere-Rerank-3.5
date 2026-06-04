-- HybridRAG schema — one table holding both dense and sparse signals.

-- Postgres has extensions — plug-ins that add new types and features.
-- `pgvector` is one such extension. Loading it unlocks:
--   - the `vector(N)` data type
--   - the `<=>` cosine-distance operator
--   - the HNSW and IVFFlat index methods

CREATE EXTENSION IF NOT EXISTS vector;  -- IF NOT EXISTS = skip if already loaded


-- The single table that holds everything we retrieve over.
CREATE TABLE IF NOT EXISTS chunks (
    -- Anywhere you say PRIMARY KEY, Postgres silently creates a B-tree index on that column.
    id           SERIAL PRIMARY KEY,       -- Creates chunks_pkey index

    -- Provenance = the origin / history of something.
    source       VARCHAR(50)  NOT NULL,    -- 'langchain' | 'anthropic' | 'openai' | 'pinecone' | 'pydantic'
    doc_id       VARCHAR(500) NOT NULL,    -- file path within source (e.g., 'tutorial/first-steps.md')
    chunk_index  INTEGER      NOT NULL,    -- position within the doc (0, 1, 2, ...)
    title        VARCHAR(500),             -- nearest heading; can be NULL for the first chunk

    -- Content
    text         TEXT NOT NULL,

    -- Dense signal: text-embedding-3-small output is 1536 dims
    embedding    vector(1536) NOT NULL,

    -- Sparse signal: auto-derived from (title + text).
    -- Postgres recomputes this whenever the row is inserted or updated — no app-side work.
    tsv          tsvector GENERATED ALWAYS AS (
                     to_tsvector('english', coalesce(title, '') || ' ' || text)
                 ) STORED,

    -- Freshness
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW(),

    -- One chunk per (source, doc, position) — re-running ingestion replaces, not duplicates.
    -- Every UNIQUE constraint silently creates a B-tree index on those columns
    UNIQUE (source, doc_id, chunk_index) -- Creates chunks_source_doc_id_chunk_index_key index
);


--- HNSW (Hierarchical Navigable Small World) on the `embedding` column.                                 
--- Graph-based vector index — search hops along edges toward the query,                                 
                                            
--- `vector_cosine_ops` = optimize for cosine distance (the `<=>` operator).                             
--- We use cosine because text embeddings are angle-based: two vectors pointing                          
--- the same direction mean the same thing regardless of magnitude.                                      
--- (Alternative: IVFFlat — partition-based, less memory, needs a `lists` knob.)  

-- HNSW vector index using cosine distance — fast approximate nearest neighbor search.
-- Dense vector search for semantic similarity
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- GIN inverted index for full-text search via the `@@` text match operator.
-- chunks_tsv_idx — the keyword-search index
-- What it speeds up: queries like "find chunks that contain the words 'cors' and 'configuration'".
-- This is the sparse half of hybrid search. 
CREATE INDEX IF NOT EXISTS chunks_tsv_idx
    ON chunks USING gin (tsv);

-- B-tree (default) for fast `WHERE source = ...` filters.
-- chunks_source_idx — the filter-by-source index
-- What it speeds up: queries like "only search inside the LangChain docs, not all 5 sources."
-- This is for metadata filtering (= equality)
CREATE INDEX IF NOT EXISTS chunks_source_idx
    ON chunks (source);


-- Two indexes were auto-created by Postgres:
-- chunks_pkey                           PRIMARY KEY, btree (id)
-- chunks_source_doc_id_chunk_index_key  UNIQUE CONSTRAINT, btree (source, doc_id, chunk_index)