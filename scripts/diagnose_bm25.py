
from src.retrieval.lexical import BM25Retriever

retriever = BM25Retriever()

questions = {
    "q001": "طبق ماده 154 قانون مجازات تعزیرات، سرقت در چه شرایطی مشمول مجازات مقرر در این ماده می‌شود؟",
    "q010": "طبق ماده ۱۵۴ قانون مجازات (تعزیرات)، یکی از شرایط پنج‌گانه سرقت چیست؟",
}

target_ids = {
    "taazirat-0130",
    "taazirat-0131",
    "taazirat-0132",
    "taazirat-0133",
    "taazirat-0136",
    "taazirat-0141",
}

for question_id, query in questions.items():

    print("\n" + "=" * 80)
    print("QUESTION:", question_id)
    print("=" * 80)

    results = retriever.search(query, top_k=20)

    for item in results:

        if item["chunk_id"] in target_ids:

            print(
                f"Rank: {item['rank']:2} | "
                f"Score: {item['score']:.6f} | "
                f"Chunk: {item['chunk_id']}"
            )
