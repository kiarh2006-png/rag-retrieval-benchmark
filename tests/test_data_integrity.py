
import json
import numpy as np
import pytest
from pathlib import Path


# Project paths
ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "data" / "corpus" / "chunks.jsonl"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "dense_embeddings.npy"
MANIFEST_PATH = ROOT / "data" / "metadata" / "manifest.json"


@pytest.fixture(scope="session")
def chunks():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="session")
def embeddings():
    return np.load(EMBEDDINGS_PATH)


@pytest.fixture(scope="session")
def manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_data_files_exist():
    assert CORPUS_PATH.exists()
    assert EMBEDDINGS_PATH.exists()
    assert MANIFEST_PATH.exists()


def test_total_chunks(chunks):
    assert len(chunks) == 5782


def test_unique_chunk_ids(chunks):
    ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_embedding_shape(embeddings):
    assert embeddings.shape == (5782, 1024)


def test_embeddings_are_finite(embeddings):
    assert np.isfinite(embeddings).all()


def test_embeddings_are_nonzero(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)

    assert np.all(norms > 0)


def test_all_chunks_have_valid_data(chunks):
    for chunk in chunks:
        assert chunk["chunk_id"]
        assert chunk["text"].strip()
        assert "document_id" in chunk["metadata"]


def test_document_count(chunks):
    document_ids = {
        chunk["metadata"]["document_id"]
        for chunk in chunks
    }

    assert len(document_ids) == 20


def test_embedding_dtype(embeddings):
    assert embeddings.dtype == np.float32


def test_manifest(manifest, chunks, embeddings):
    assert manifest["total_chunks"] == len(chunks)
    assert manifest["embedding_dimension"] == embeddings.shape[1]
    assert manifest["embedding_recomputed"] is False
