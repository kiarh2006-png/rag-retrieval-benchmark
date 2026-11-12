
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
OUTPUT_PATH = PROJECT_ROOT / "data/eval/benchmark_results.json"


def load_questions():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def print_summary(name, summary):
    print(f"\n{name}")
    print("=" * 65)

    print("Total queries:", summary["total_queries"])
    print("Answerable:", summary["answerable_queries"])
    print("Unanswerable:", summary["unanswerable_queries"])

    for metric in [
        "hit@1",
        "hit@3",
        "hit@5",
        "recall@5",
        "mrr",
        "ndcg@5",
    ]:
        print(f"{metric:12}: {summary[metric]:.4f}")


def main():
    questions = load_questions()

    print("QUESTIONS:", len(questions))
    print("Initializing retrievers...")

    # Initialize each retrieval engine only once.
    hybrid = HybridRetriever(rrf_k=60)

    dense = hybrid.dense
    bm25 = hybrid.bm25

    retrievers = {
        "Dense": dense,
        "BM25": bm25,
        "Hybrid": hybrid,
    }

    all_results = {}

    for name, retriever in retrievers.items():

        print(f"\nRunning {name}...")

        query_results = []

        for question in questions:

            query = question["question"]

            if name == "Hybrid":
                results = retriever.search(
                    query,
                    top_k=5,
                    candidate_k=100,
                )
            else:
                results = retriever.search(
                    query,
                    top_k=5,
                )

            retrieved_ids = [
                item["chunk_id"]
                for item in results
            ]

            metrics = evaluate_query(
                retrieved_ids,
                question["relevant_chunk_ids"],
                answerable=question["answerable"],
            )

            query_results.append({
                "question_id": question["id"],
                "category": question["category"],
                "answerable": question["answerable"],
                "retrieved_chunk_ids": retrieved_ids,
                "metrics": metrics,
            })

        summary = aggregate_results([
            item["metrics"] for item in query_results
        ])

        all_results[name] = {
            "summary": summary,
            "queries": query_results,
        }

        print_summary(name, summary)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
