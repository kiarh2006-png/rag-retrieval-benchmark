import json
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def test_corpus_and_embeddings_alignment():
    chunks = [json.loads(x) for x in (ROOT/"data/corpus/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    emb = np.load(ROOT/"data/embeddings/dense_embeddings.npy", mmap_mode="r")
    manifest = json.loads((ROOT/"data/metadata/manifest.json").read_text(encoding="utf-8"))
    assert len(chunks) == 5782
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert emb.shape == (len(chunks), 1024)
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all()
    assert manifest["embedding_recomputed"] is False
