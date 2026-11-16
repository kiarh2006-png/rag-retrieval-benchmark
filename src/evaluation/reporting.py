"""Category summaries and reproducible failure-case exports."""
from collections import defaultdict
from src.evaluation.metrics import aggregate_results


def summarize_by_category(query_rows):
    groups = defaultdict(list)
    for row in query_rows:
        groups[row["category"]].append(row["metrics"])
    return {
        category: aggregate_results(metrics)
        for category, metrics in sorted(groups.items())
    }


def build_failure_candidates(method_results, questions_by_id, chunk_lookup, limit=1000):
    """Export misses without inventing a root cause; human review is required."""
    failures = []
    for method, payload in method_results.items():
        for row in payload["queries"]:
            q = questions_by_id[row["question_id"]]
            if not q["answerable"]:
                continue
            m = row["metrics"]
            if m["hit@5"] == 1 and m["recall@5"] == 1.0:
                continue
            failures.append({
                "question_id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "retrieval_method": method,
                "expected_relevant_chunks": q["relevant_chunk_ids"],
                "retrieved_chunks": [
                    {"chunk_id": cid, "text": chunk_lookup.get(cid, {}).get("text", "")[:1200]}
                    for cid in row["retrieved_chunk_ids"]
                ],
                "metrics": m,
                "failure_type_candidate": (
                    "relevant_chunk_ranked_low_or_missing"
                    if m["hit@5"] else "no_relevant_chunk_in_top5"
                ),
                "root_cause": "REQUIRES_MANUAL_REVIEW",
                "possible_fix": "REQUIRES_MANUAL_REVIEW",
            })
    failures.sort(key=lambda x: (
        x["metrics"]["hit@5"], x["metrics"]["recall@5"],
        x["metrics"]["mrr"], x["question_id"], x["retrieval_method"]
    ))
    return failures[:limit]
