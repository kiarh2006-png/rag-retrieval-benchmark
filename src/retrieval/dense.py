"""Dense cosine retrieval over the frozen BGE-M3 corpus embeddings."""
import json
import urllib.request
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "corpus" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "dense_embeddings.npy"
OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "bge-m3"
_QUERY_EMBED_CACHE = {}


class DenseRetriever:
    def __init__(self, query_embedder=None, chunks=None, embeddings=None):
        self.chunks = chunks if chunks is not None else self._load_chunks()
        self.embeddings = embeddings if embeddings is not None else np.load(EMBEDDINGS_PATH, mmap_mode="r")
        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError("Chunks count does not match embeddings count")
        self.dimension = self.embeddings.shape[1]
        self.query_embedder = query_embedder

    def _load_chunks(self):
        with CHUNKS_PATH.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    def _embed_query(self, query):
        cache_key = (EMBEDDING_MODEL, query)
        if self.query_embedder is None and cache_key in _QUERY_EMBED_CACHE:
            return _QUERY_EMBED_CACHE[cache_key]
        if self.query_embedder is not None:
            vector = np.asarray(self.query_embedder(query), dtype=np.float32)
        else:
            payload = json.dumps({"model": EMBEDDING_MODEL, "input": query}, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(OLLAMA_URL, data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            vector = np.asarray(result["embeddings"][0], dtype=np.float32)
        if vector.shape != (self.dimension,):
            raise ValueError(f"Query dimension {vector.shape} does not match {self.dimension}")
        if not np.isfinite(vector).all():
            raise ValueError("Query embedding contains invalid values")
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Query embedding has zero norm")
        vector = vector / norm
        if self.query_embedder is None:
            _QUERY_EMBED_CACHE[cache_key] = vector
        return vector

    def search(self, query, top_k=5):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vector = self._embed_query(query)
        scores = self.embeddings @ vector
        # stable tie-breaking by original corpus position
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [{
            "rank": rank, "chunk_id": self.chunks[int(i)]["chunk_id"],
            "text": self.chunks[int(i)]["text"], "metadata": self.chunks[int(i)]["metadata"],
            "score": float(scores[i]),
        } for rank, i in enumerate(indices, start=1)]
