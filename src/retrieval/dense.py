
import json
import urllib.request
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = PROJECT_ROOT / "data" / "corpus" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "dense_embeddings.npy"

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "bge-m3"


class DenseRetriever:

    def __init__(self):
        self.chunks = self._load_chunks()

        self.embeddings = np.load(
            EMBEDDINGS_PATH,
            mmap_mode="r"
        )

        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(
                f"Chunks count ({len(self.chunks)}) "
                f"does not match embeddings count "
                f"({self.embeddings.shape[0]})"
            )

        self.dimension = self.embeddings.shape[1]

    def _load_chunks(self):
        chunks = []

        with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
            for line in file:
                chunks.append(json.loads(line))

        return chunks

    def _embed_query(self, query):

        payload = {
            "model": EMBEDDING_MODEL,
            "input": query
        }

        data = json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8")

        request = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

        vector = np.asarray(
            result["embeddings"][0],
            dtype=np.float32
        )

        if vector.shape != (self.dimension,):
            raise ValueError(
                f"Query dimension {vector.shape} "
                f"does not match document dimension "
                f"{self.dimension}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "Query embedding contains invalid values"
            )

        norm = np.linalg.norm(vector)

        if norm == 0:
            raise ValueError(
                "Query embedding has zero norm"
            )

        return vector / norm

    def search(self, query, top_k=5):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = self._embed_query(query)

        scores = self.embeddings @ query_vector

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, index in enumerate(top_indices, start=1):

            chunk = self.chunks[int(index)]

            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": float(scores[index])
            })

        return results
