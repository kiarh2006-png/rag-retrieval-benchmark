
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever
from src.evaluation.metrics import (
    evaluate_query,
    aggregate_results,
)


GROUND_TRUTH_PATH = PROJECT_ROOT / "data/eval/ground_truth.json"
OUTPUT_PATH = PROJECT_ROOT / "data/eval/weighted_comparison.json"

RRF_K = 60
CANDIDATE_K = 100
TOP_K = 5


def load_questions():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def weighted_rrf(dense_results, bm25_results, dense_weight=2, bm25_weight=1):

    fused = {}

    for results, weight in (
        (dense_results, dense_weight),
        (bm25_results, bm25_weight),
    ):
        for item in results:

            chunk_id = item["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = 0.0

            fused[chunk_id] += (
                weight / (RRF_K + item["rank"])
            )

    ranked = sorted(
        fused.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [chunk_id for chunk_id, score in ranked[:TOP_K]]


def main():

    questions = load_questions()

    print("QUESTIONS:", len(questions))
    print("Initializing retrievers...")

    hybrid = HybridRetriever(rrf_k=RRF_K)

    dense = hybrid.dense
    bm25 = hybrid.bm25

    methods = {
        "Dense": [],
        "BM25": [],
        "Hybrid_Standard": [],
        "Hybrid_Dense_Weighted": [],
    }

    for index, question in enumerate(questions, start=1):

        query = question["question"]

        print(f"Processing {index}/{len(questions)}: {question['id']}")

        dense_results = dense.search(
            query,
            top_k=CANDIDATE_K,
        )

        bm25_results = bm25.search(
            query,
            top_k=CANDIDATE_K,
        )

        standard_results = hybrid.search(
            query,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
        )

        dense_ids = [
            item["chunk_id"]
            for item in dense_results[:TOP_K]
        ]

        bm25_ids = [
            item["chunk_id"]
            for item in bm25_results[:TOP_K]
        ]

        standard_ids = [
            item["chunk_id"]
            for item in standard_results
        ]

        weighted_ids = weighted_rrf(
            dense_results,
            bm25_results,
            dense_weight=2,
            bm25_weight=1,
        )

        retrieved_methods = {
            "Dense": dense_ids,
            "BM25": bm25_ids,
            "Hybrid_Standard": standard_ids,
            "Hybrid_Dense_Weighted": weighted_ids,
        }

        for name, retrieved_ids in retrieved_methods.items():

            metrics = evaluate_query(
                retrieved_ids,
                question["relevant_chunk_ids"],
                answerable=question["answerable"],
            )

            methods[name].append({
                "question_id": question["id"],
                "metrics": metrics,
                "retrieved_chunk_ids": retrieved_ids,
            })

    all_results = {}

    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    for name, query_results in methods.items():

        summary = aggregate_results([
            item["metrics"]
            for item in query_results
        ])

        all_results[name] = {
            "summary": summary,
            "queries": query_results,
        }

        print("\n", name)
        print("-" * 50)

        for metric in (
            "hit@1",
            "hit@3",
            "hit@5",
            "recall@5",
            "mrr",
            "ndcg@5",
        ):
            print(f"{metric:12}: {summary[metric]:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nCOMPARISON COMPLETED")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
