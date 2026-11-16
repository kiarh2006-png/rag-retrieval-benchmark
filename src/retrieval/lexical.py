"""Persian-normalized BM25 retrieval with a dependency-free fallback."""
import json
import math
import re
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # lets unit tests and offline installs work
    BM25Okapi = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "corpus" / "chunks.jsonl"


def normalize(text):
    text = str(text)
    replacements = {"ي":"ی", "ك":"ک", "ۀ":"ه", "ة":"ه", "أ":"ا", "إ":"ا", "ؤ":"و", "ئ":"ی"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[۰-۹]", lambda m: str("۰۱۲۳۴۵۶۷۸۹".index(m.group())), text)
    text = re.sub(r"[٠-٩]", lambda m: str("٠١٢٣٤٥٦٧٨٩".index(m.group())), text)
    return text


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9آ-ی]+", normalize(text).lower())


class _FallbackBM25:
    """Small Okapi BM25 implementation used only when rank_bm25 is absent."""
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.lengths) / max(self.n, 1)
        self.df = {}
        self.tf = []
        for doc in corpus:
            counts = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self.tf.append(counts)
            for token in counts:
                self.df[token] = self.df.get(token, 0) + 1

    def get_scores(self, query_tokens):
        scores = [0.0] * self.n
        for token in query_tokens:
            df = self.df.get(token, 0)
            if not df:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, counts in enumerate(self.tf):
                freq = counts.get(token, 0)
                if freq:
                    denom = freq + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avgdl, 1))
                    scores[i] += idf * freq * (self.k1 + 1) / denom
        return scores


class BM25Retriever:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else self._load_chunks()
        self.tokenized_corpus = [tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = (BM25Okapi(self.tokenized_corpus)
                     if BM25Okapi is not None
                     else _FallbackBM25(self.tokenized_corpus))

    def _load_chunks(self):
        with CHUNKS_PATH.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    def search(self, query, top_k=5):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        tokens = tokenize(query)
        if not tokens:
            raise ValueError("Query contains no searchable tokens")
        scores = self.bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        return [{
            "rank": rank,
            "chunk_id": self.chunks[i]["chunk_id"],
            "text": self.chunks[i]["text"],
            "metadata": self.chunks[i]["metadata"],
            "score": float(scores[i]),
        } for rank, i in enumerate(indices, start=1)]
