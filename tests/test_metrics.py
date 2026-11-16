import pytest
from src.evaluation.metrics import evaluate_query, aggregate_results, validate_ground_truth


def test_hit_at_k_and_recall():
    r = evaluate_query(["x", "a", "b", "c", "d"], ["a", "b", "z"])
    assert r["hit@1"] == 0
    assert r["hit@3"] == 1
    assert r["hit@5"] == 1
    assert r["recall@5"] == pytest.approx(2/3)


def test_mrr():
    assert evaluate_query(["x", "a"], ["a"])["mrr"] == pytest.approx(0.5)
    assert evaluate_query(["x", "y"], ["a"])["mrr"] == 0.0


def test_ndcg_at_5():
    r = evaluate_query(["x", "a", "b"], ["a", "b"])
    expected = (1 / __import__("math").log2(3) + 1 / __import__("math").log2(4)) / (1 + 1 / __import__("math").log2(3))
    assert r["ndcg@5"] == pytest.approx(expected)


def test_unanswerable_is_excluded_from_ranking_metrics():
    r = evaluate_query(["x", "y"], [], answerable=False)
    assert r["evaluated"] is False
    assert r["hit@5"] is None
    s = aggregate_results([r])
    assert s["unanswerable_queries"] == 1
    assert s["mrr"] is None


@pytest.mark.parametrize("ids,answerable", [
    (["x", "x"], True), ([], True), (["x"], False), ([""], True),
])
def test_invalid_ground_truth(ids, answerable):
    with pytest.raises(ValueError):
        validate_ground_truth(ids, answerable)


def test_retrieved_duplicate_rejected():
    with pytest.raises(ValueError):
        evaluate_query(["x", "x"], ["x"])
