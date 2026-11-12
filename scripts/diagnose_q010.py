
from src.retrieval.hybrid import HybridRetriever


QUERY = "طبق ماده ۱۵۴ قانون مجازات (تعزیرات)، یکی از شرایط پنج‌گانه سرقت چیست؟"
TARGET = "taazirat-0131"

retriever = HybridRetriever(rrf_k=60)

dense_results = retriever.dense.search(QUERY, top_k=100)
bm25_results = retriever.bm25.search(QUERY, top_k=100)

hybrid_results = retriever.search(
    QUERY,
    top_k=100,
    candidate_k=100
)


def inspect(name, results):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    target = next(
        (item for item in results if item["chunk_id"] == TARGET),
        None
    )

    if target:
        print("TARGET FOUND")
        print("Rank:", target.get("rank"))
        print(
            "Score:",
            target.get("score", target.get("rrf_score"))
        )
    else:
        print("TARGET NOT FOUND IN RESULTS")

    print("\nTOP 15:")

    for item in results[:15]:
        print(
            "Rank:", item.get("rank"),
            "| ID:", item["chunk_id"],
            "| Score:", item.get("score", item.get("rrf_score"))
        )


inspect("DENSE", dense_results)
inspect("BM25", bm25_results)
inspect("HYBRID", hybrid_results)
