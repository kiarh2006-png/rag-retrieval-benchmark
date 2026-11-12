
from src.retrieval.dense import DenseRetriever
from src.retrieval.lexical import BM25Retriever


class HybridRetriever:

    def __init__(self, rrf_k=60):
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")

        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()
        self.rrf_k = rrf_k

    def search(self, query, top_k=5, candidate_k=100):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k < 1 or candidate_k < 1:
            raise ValueError("top_k and candidate_k must be positive")

        dense_results = self.dense.search(
            query,
            top_k=candidate_k
        )

        lexical_results = self.bm25.search(
            query,
            top_k=candidate_k
        )

        fused = {}

        for results in (dense_results, lexical_results):
            for item in results:
                chunk_id = item["chunk_id"]

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": item["text"],
                        "metadata": item["metadata"],
                        "rrf_score": 0.0
                    }

                fused[chunk_id]["rrf_score"] += (
                    1.0 / (self.rrf_k + item["rank"])
                )

        ranked = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True
        )

        results = []

        for rank, item in enumerate(ranked[:top_k], start=1):
            results.append({
                **item,
                "rank": rank
            })

        return results
