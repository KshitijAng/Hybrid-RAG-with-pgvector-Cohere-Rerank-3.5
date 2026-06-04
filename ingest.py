"""Bulk ingest the corpus into Postgres.

Walks data/corpus/<source>/*.md, chunks each file, embeds in batches, and
upserts rows into the chunks table.

Idempotent: re-running upserts via the (source, doc_id, chunk_index) unique
key — existing rows get updated, no duplicates.

Run:
    python ingest.py                       # full corpus
    python ingest.py --source anthropic    # one source only (good for testing)
"""

import argparse
import time

from chunk import CORPUS_DIR, chunk_file
from db import connect
from embed import EMBED_BATCH_SIZE, embed_texts


# Sleep between embedding batches to stay under GitHub Models' 500K TPM.
# Each batch ≈ 80K tokens → ~6 batches/min ceiling.
SECONDS_BETWEEN_BATCHES = 5


INSERT_SQL = """
    INSERT INTO chunks (source, doc_id, chunk_index, title, text, embedding)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (source, doc_id, chunk_index)
    DO UPDATE SET
        title      = EXCLUDED.title,
        text       = EXCLUDED.text,
        embedding  = EXCLUDED.embedding,
        updated_at = NOW()
"""


def collect_chunks(only_source: str | None = None) -> list[dict]:
    """Walk the corpus folder and return every chunk as a flat list."""
    chunks: list[dict] = []
    sources = sorted(d for d in CORPUS_DIR.iterdir() if d.is_dir())
    if only_source:
        sources = [d for d in sources if d.name == only_source]

    for source_dir in sources:
        before = len(chunks)
        for f in sorted(source_dir.glob("*.md")):
            chunks.extend(chunk_file(source_dir.name, f.name))
        print(f"  {source_dir.name:<14} {len(chunks) - before} chunks")
    return chunks


def main(only_source: str | None = None) -> None:
    print("=== Step 1: chunking ===")
    chunks = collect_chunks(only_source)
    total = len(chunks)
    print(f"  TOTAL          {total} chunks\n")

    print("=== Step 2: embed + insert (5s pacing between batches) ===")
    with connect() as conn:
        with conn.cursor() as cur:
            for i in range(0, total, EMBED_BATCH_SIZE):
                batch = chunks[i : i + EMBED_BATCH_SIZE]
                texts = [c["text"] for c in batch]

                t0 = time.time()
                vectors = embed_texts(texts)
                t_embed = time.time() - t0

                cur.executemany(
                    INSERT_SQL,
                    [
                        (
                            c["source"],
                            c["doc_id"],
                            c["chunk_index"],
                            c["title"],
                            c["text"],
                            v,
                        )
                        for c, v in zip(batch, vectors)
                    ],
                )
                conn.commit()

                done = min(i + EMBED_BATCH_SIZE, total)
                print(f"  [{done:>5}/{total}]  embed {t_embed:>4.1f}s  inserted {len(batch)} rows")

                if i + EMBED_BATCH_SIZE < total:
                    time.sleep(SECONDS_BETWEEN_BATCHES)

    print(f"\nDone — {total} chunks in Postgres.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Only ingest this one source (e.g., 'anthropic')")
    args = parser.parse_args()
    main(only_source=args.source)
