"""Retrieval-only eval: recall@5 and MRR across all 4 retrieval modes.

Loads data/eval_queries.json (30 labeled queries with expected_doc_ids[]) and
runs each one through dense / sparse / hybrid / hybrid_rerank.

  recall@5 = fraction of queries where any expected doc is in the top-5
  MRR      = mean of (1 / rank of first matching expected doc) — 0 if not found

Each query is embedded ONCE and the vector is cached to disk (eval/query_embeddings.json)
so reruns are free. Per-query results are checkpointed to eval/results_retrieval.json
after every query — a mid-run rate-limit doesn't lose work; re-run resumes.

Run:
    ./venv/bin/python eval/retrieval_eval.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

# Make the parent dir importable so we can pull retrieve/embed without a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embed import embed_texts
from retrieve import dense, hybrid, hybrid_rerank, sparse


ROOT = Path(__file__).resolve().parent.parent
EVAL_QUERIES = ROOT / "data" / "eval_queries.json"
EMBED_CACHE = ROOT / "eval" / "query_embeddings.json"
RESULTS_PATH = ROOT / "eval" / "results_retrieval.json"

TOP_K = 5
COHERE_PACE_SEC = 6.5  # Cohere free tier rerank: ~10/min — pace between queries


def doc_id_of(row: dict) -> str:
    return f"{row['source']}/{row['doc_id']}"


def reciprocal_rank(top_docs: list[str], expected: set[str], k: int = TOP_K) -> tuple[float, float]:
    """Return (recall@k, reciprocal_rank). 1.0 recall if any expected doc lands in top-k."""
    for rank, did in enumerate(top_docs[:k], start=1):
        if did in expected:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


def get_query_embedding(query: str, cache: dict) -> list[float]:
    """Return embedding for `query`, caching to disk so reruns don't burn quota."""
    if query in cache:
        return cache[query]
    [vec] = embed_texts([query])
    cache[query] = vec
    EMBED_CACHE.write_text(json.dumps(cache))
    return vec


def run_one(query: str, query_vec: list[float]) -> dict:
    """Run all 4 modes for one query. We pass the precomputed embedding into the
    retrievers via a monkey-patched embed_texts wrapper so the cache is honored."""
    per_mode = {}
    for mode_name, fn in (("dense", dense), ("sparse", sparse), ("hybrid", hybrid), ("hybrid_rerank", hybrid_rerank)):
        t0 = time.time()
        if mode_name == "sparse":
            results = fn(query, k=TOP_K)
        else:
            # dense/hybrid/hybrid_rerank embed internally; we override by stuffing the cache.
            # Cheapest path: patch embed_texts for the duration of this call.
            import embed as embed_mod
            original = embed_mod.embed_texts
            embed_mod.embed_texts = lambda texts: [query_vec] if texts == [query] else original(texts)
            try:
                results = fn(query, k=TOP_K)
            finally:
                embed_mod.embed_texts = original
        per_mode[mode_name] = {
            "top_docs": [doc_id_of(r) for r in results[:TOP_K]],
            "latency_ms": int((time.time() - t0) * 1000),
        }
    return per_mode


def main() -> None:
    eval_set = json.loads(EVAL_QUERIES.read_text())
    n = len(eval_set)

    # Load caches (embeddings + per-query results).
    embed_cache = json.loads(EMBED_CACHE.read_text()) if EMBED_CACHE.exists() else {}
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}

    if results:
        print(f"Resuming — {len(results)}/{n} queries already scored.")

    for i, item in enumerate(eval_set, 1):
        q = item["query"]
        if q in results:
            continue
        print(f"\n[{i:>2}/{n}] {q[:80]}{'...' if len(q) > 80 else ''}")
        qvec = get_query_embedding(q, embed_cache)
        per_mode = run_one(q, qvec)
        results[q] = {"expected_doc_ids": item["expected_doc_ids"], "modes": per_mode}
        for mode in ("dense", "sparse", "hybrid", "hybrid_rerank"):
            r = per_mode[mode]
            recall, rr = reciprocal_rank(r["top_docs"], set(item["expected_doc_ids"]))
            top = r["top_docs"][0] if r["top_docs"] else "(none)"
            mark = "✓" if recall else "✗"
            print(f"   {mode:<14} {mark}  rr={rr:.2f}  {r['latency_ms']:>5}ms  top: {top}")
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        if i < n:
            time.sleep(COHERE_PACE_SEC)

    # Aggregate.
    agg = {m: {"recall": [], "rr": [], "latency": []} for m in ("dense", "sparse", "hybrid", "hybrid_rerank")}
    for item in eval_set:
        rec = results[item["query"]]
        expected = set(rec["expected_doc_ids"])
        for mode in agg:
            top = rec["modes"][mode]["top_docs"]
            recall, rr = reciprocal_rank(top, expected)
            agg[mode]["recall"].append(recall)
            agg[mode]["rr"].append(rr)
            agg[mode]["latency"].append(rec["modes"][mode]["latency_ms"])

    print("\n" + "=" * 76)
    print(f"{'Mode':<16}{'recall@5':>12}{'MRR':>10}{'avg latency':>18}")
    print("-" * 76)
    for mode in agg:
        r = statistics.mean(agg[mode]["recall"])
        mrr = statistics.mean(agg[mode]["rr"])
        lat = statistics.mean(agg[mode]["latency"])
        print(f"{mode:<16}{r:>12.3f}{mrr:>10.3f}{lat:>15.0f}ms")
    print("=" * 76)

    # Failures on the best mode — useful debug.
    best = max(agg, key=lambda m: statistics.mean(agg[m]["recall"]))
    misses = [(i + 1, eval_set[i]["query"]) for i in range(n) if agg[best]["recall"][i] == 0.0]
    if misses:
        print(f"\n{best} missed {len(misses)}/{n}:")
        for idx, q in misses:
            print(f"  [{idx}] {q}")


if __name__ == "__main__":
    main()
