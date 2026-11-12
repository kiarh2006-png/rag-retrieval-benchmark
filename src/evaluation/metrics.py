
import math


def validate_ground_truth(relevant_chunk_ids, answerable):
    if not isinstance(answerable, bool):
        raise ValueError("answerable must be boolean")

    if not isinstance(relevant_chunk_ids, list):
        raise ValueError("relevant_chunk_ids must be a list")

    if answerable and not relevant_chunk_ids:
        raise ValueError("Answerable questions must have relevant chunks")

    if not answerable and relevant_chunk_ids:
        raise ValueError("Unanswerable questions must not have relevant chunks")

    if len(relevant_chunk_ids) != len(set(relevant_chunk_ids)):
        raise ValueError("Duplicate relevant chunk IDs")

    if any(not isinstance(x, str) or not x.strip() for x in relevant_chunk_ids):
        raise ValueError("Invalid relevant chunk ID")


def evaluate_query(retrieved_chunk_ids, relevant_chunk_ids, answerable=True):
    validate_ground_truth(relevant_chunk_ids, answerable)

    if not answerable:
        return {
            "answerable": False,
            "evaluated": False,
            "hit@1": None,
            "hit@3": None,
            "hit@5": None,
            "recall@5": None,
            "mrr": None,
            "ndcg@5": None,
        }

    relevant = set(relevant_chunk_ids)

    def hit(k):
        return int(bool(relevant.intersection(retrieved_chunk_ids[:k])))

    top_5 = retrieved_chunk_ids[:5]
    relevant_found = len(set(top_5) & relevant)

    recall = relevant_found / len(relevant)

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(top_5, start=1)
        if chunk_id in relevant
    )

    ideal_relevant = min(len(relevant), 5)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )

    return {
        "answerable": True,
        "evaluated": True,
        "hit@1": hit(1),
        "hit@3": hit(3),
        "hit@5": hit(5),
        "recall@5": recall,
        "mrr": reciprocal_rank,
        "ndcg@5": dcg / idcg if idcg else 0.0,
    }


def aggregate_results(results):
    answerable_results = [
        result for result in results
        if result["evaluated"]
    ]

    unanswerable_count = sum(
        1 for result in results
        if not result["answerable"]
    )

    metrics = [
        "hit@1", "hit@3", "hit@5",
        "recall@5", "mrr", "ndcg@5"
    ]

    summary = {
        "total_queries": len(results),
        "answerable_queries": len(answerable_results),
        "unanswerable_queries": unanswerable_count,
    }

    for metric in metrics:
        summary[metric] = (
            sum(result[metric] for result in answerable_results)
            / len(answerable_results)
            if answerable_results else 0.0
        )

    return summary
