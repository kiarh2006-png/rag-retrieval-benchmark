
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "corpus" / "chunks.jsonl"


def normalize(text):
    text = str(text)

    replacements = {
        "ي": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "أ": "ا",
        "إ": "ا",
        "ؤ": "و",
        "ئ": "ی",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[۰-۹]", lambda m: str("۰۱۲۳۴۵۶۷۸۹".index(m.group())), text)
    text = re.sub(r"[٠-٩]", lambda m: str("٠١٢٣٤٥٦٧٨٩".index(m.group())), text)

    return text


def tokenize(text):
    text = normalize(text).lower()
    return re.findall(r"[a-zA-Z0-9آ-ی]+", text)


class BM25Retriever:

    def __init__(self):
        self.chunks = self._load_chunks()

        self.tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _load_chunks(self):
        chunks = []

        with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
            for line in file:
                chunks.append(json.loads(line))

        return chunks

    def search(self, query, top_k=5):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        tokens = tokenize(query)

        if not tokens:
            raise ValueError("Query contains no searchable tokens")

        scores = self.bm25.get_scores(tokens)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for rank, index in enumerate(top_indices, start=1):

            chunk = self.chunks[index]

            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": float(scores[index])
            })

        return results
