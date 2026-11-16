import pytest
from src.retrieval.hybrid import rrf_fuse


def test_rrf_fuses_two_ranked_lists():
    a = [{"rank": 1, "chunk_id": "a"}, {"rank": 2, "chunk_id": "b"}]
    b = [{"rank": 1, "chunk_id": "b"}, {"rank": 2, "chunk_id": "c"}]
    out = rrf_fuse([a, b], rrf_k=60)
    assert out[0]["chunk_id"] in {"a", "b"}
    scores = {x["chunk_id"]: x["rrf_score"] for x in out}
    assert scores["b"] == pytest.approx(1/62 + 1/61)


def test_rrf_deduplicates_same_chunk_within_and_across_lists():
    a = [{"rank": 1, "chunk_id": "a"}, {"rank": 2, "chunk_id": "a"}]
    b = [{"rank": 1, "chunk_id": "a"}]
    out = rrf_fuse([a, b], rrf_k=60)
    assert [x["chunk_id"] for x in out] == ["a"]
    assert out[0]["rrf_score"] == pytest.approx(2/61)


def test_rrf_top_k_and_invalid_values():
    with pytest.raises(ValueError):
        rrf_fuse([[{"rank": 1, "chunk_id": "a"}]], rrf_k=0)
    assert len(rrf_fuse([[{"rank": 1, "chunk_id": "a"}]], top_k=1)) == 1
