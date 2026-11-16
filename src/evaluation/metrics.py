"""Ranking metrics for chunk-level retrieval evaluation."""
import math

METRICS = ("hit@1", "hit@3", "hit@5", "recall@5", "mrr", "ndcg@5")


def validate_ground_truth(relevant_chunk_ids, answerable):
    if not isinstance(answerable, bool):
        raise ValueError("answerable must be boolean")
    if not isinstance(relevant_chunk_ids, list):
        raise ValueError("relevant_chunk_ids must be a list")
    if any(not isinstance(x, str) or not x.strip() for x in relevant_chunk_ids):
        raise ValueError("Invalid relevant chunk ID")
    if len(relevant_chunk_ids) != len(set(relevant_chunk_ids)):
        raise ValueError("Duplicate relevant chunk IDs")
    if answerable and not relevant_chunk_ids:
        raise ValueError("Answerable questions must have relevant chunks")
    if not answerable and relevant_chunk_ids:
        raise ValueError("Unanswerable questions must not have relevant chunks")


def evaluate_query(retrieved_chunk_ids, relevant_chunk_ids, answerable=True):
    validate_ground_truth(relevant_chunk_ids, answerable)
    if not isinstance(retrieved_chunk_ids, list):
        raise ValueError("retrieved_chunk_ids must be a list")
    if len(retrieved_chunk_ids) != len(set(retrieved_chunk_ids)):
        raise ValueError("Retrieved chunk IDs must be unique")

    if not answerable:
        return {
            "answerable": False, "evaluated": False,
            **{metric: None for metric in METRICS},
        }

    relevant = set(relevant_chunk_ids)
    top5 = retrieved_chunk_ids[:5]
    found = len(set(top5) & relevant)
    first_rank = next(
        (rank for rank, cid in enumerate(retrieved_chunk_ids, start=1)
         if cid in relevant), None
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(top5, start=1) if cid in relevant
    )
    ideal_count = min(len(relevant), 5)
    idcg = sum(1.0 / math.log2(rank + 1)
               for rank in range(1, ideal_count + 1))

    return {
        "answerable": True, "evaluated": True,
        "hit@1": int(bool(set(retrieved_chunk_ids[:1]) & relevant)),
        "hit@3": int(bool(set(retrieved_chunk_ids[:3]) & relevant)),
        "hit@5": int(bool(set(top5) & relevant)),
        "recall@5": found / len(relevant),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg@5": dcg / idcg if idcg else 0.0,
    }


def aggregate_results(results):
    valid = [r for r in results if r["evaluated"]]
    return {
        "total_queries": len(results),
        "answerable_queries": len(valid),
        "unanswerable_queries": sum(not r["answerable"] for r in results),
        **{
            metric: (sum(r[metric] for r in valid) / len(valid) if valid else None)
            for metric in METRICS
        },
    }
