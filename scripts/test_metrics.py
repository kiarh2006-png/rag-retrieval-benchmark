
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.evaluation.metrics import (
    evaluate_query,
    aggregate_results,
    validate_ground_truth,
)


def test_metrics():
    result = evaluate_query(
        ["a", "b", "c", "d", "e"],
        ["b", "d"],
    )

    assert result["hit@1"] == 0
    assert result["hit@3"] == 1
    assert result["hit@5"] == 1
    assert result["recall@5"] == 1.0
    assert result["mrr"] == 0.5
    assert result["ndcg@5"] > 0


def test_unanswerable():
    result = evaluate_query(
        ["a", "b", "c"],
        [],
        answerable=False,
    )

    assert result["evaluated"] is False
    assert result["hit@5"] is None


def test_invalid_ground_truth():
    with pytest.raises(ValueError):
        validate_ground_truth([], True)


def test_duplicate_ground_truth():
    with pytest.raises(ValueError):
        validate_ground_truth(["a", "a"], True)


def test_aggregate_results():
    results = [
        evaluate_query(["a", "b"], ["a"]),
        evaluate_query(["b", "a"], ["a"]),
        evaluate_query(["a", "b"], [], answerable=False),
    ]

    summary = aggregate_results(results)

    assert summary["total_queries"] == 3
    assert summary["answerable_queries"] == 2
    assert summary["unanswerable_queries"] == 1
    assert summary["hit@1"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
