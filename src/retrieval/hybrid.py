"""Reciprocal Rank Fusion and hybrid retrieval."""
from src.retrieval.dense import DenseRetriever
from src.retrieval.lexical import BM25Retriever


def rrf_fuse(result_lists, rrf_k=60, weights=None, top_k=None):
    """Fuse ranked lists by chunk_id; duplicates are merged into one result."""
    if not isinstance(rrf_k, int) or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer")
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists) or any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative and match result lists")

    fused = {}
    for results, weight in zip(result_lists, weights):
        seen_in_list = set()
        for fallback_rank, item in enumerate(results, start=1):
            cid = item["chunk_id"]
            if cid in seen_in_list:
                continue
            seen_in_list.add(cid)
            rank = int(item.get("rank", fallback_rank))
            if rank < 1:
                raise ValueError("rank must be >= 1")
            if cid not in fused:
                fused[cid] = {
                    "chunk_id": cid,
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                    "rrf_score": 0.0,
                }
            fused[cid]["rrf_score"] += float(weight) / (rrf_k + rank)

    ranked = sorted(fused.values(), key=lambda x: (-x["rrf_score"], x["chunk_id"]))
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        ranked = ranked[:top_k]
    return [{**item, "rank": i} for i, item in enumerate(ranked, start=1)]


class HybridRetriever:
    def __init__(self, rrf_k=60, dense=None, bm25=None, dense_weight=1.0, bm25_weight=1.0):
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        self.dense = dense or DenseRetriever()
        self.bm25 = bm25 or BM25Retriever()
        self.rrf_k = rrf_k
        self.weights = [dense_weight, bm25_weight]

    def search(self, query, top_k=5, candidate_k=100):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1 or candidate_k < 1:
            raise ValueError("top_k and candidate_k must be positive")
        dense_results = self.dense.search(query, top_k=candidate_k)
        lexical_results = self.bm25.search(query, top_k=candidate_k)
        return rrf_fuse(
            [dense_results, lexical_results],
            rrf_k=self.rrf_k,
            weights=self.weights,
            top_k=top_k,
        )
