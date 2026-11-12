
from src.retrieval.lexical import BM25Retriever, tokenize

retriever = BM25Retriever()

query = (
    "طبق ماده ۱۵۴ قانون مجازات (تعزیرات)، "
    "یکی از شرایط پنج‌گانه سرقت چیست؟"
)

target_ids = [
    "taazirat-0130",
    "taazirat-0131",
    "taazirat-0133",
    "taazirat-0141",
]

query_tokens = tokenize(query)
bm25 = retriever.bm25

print("QUERY TOKENS:", query_tokens)
print("Average document length:", bm25.avgdl)
print("BM25 k1:", bm25.k1)
print("BM25 b:", bm25.b)

for chunk_id in target_ids:

    index = next(
        i for i, chunk in enumerate(retriever.chunks)
        if chunk["chunk_id"] == chunk_id
    )

    frequencies = bm25.doc_freqs[index]
    doc_len = bm25.doc_len[index]

    print("\n" + "=" * 65)
    print("CHUNK:", chunk_id)
    print("DOCUMENT LENGTH:", doc_len)

    total = 0.0

    for token in query_tokens:
        freq = frequencies.get(token, 0)

        raw_idf = bm25.idf.get(token, 0.0)
        effective_idf = raw_idf

        if raw_idf < 0:
            effective_idf = bm25.epsilon * bm25.average_idf

        denominator = (
            freq
            + bm25.k1 * (
                1 - bm25.b
                + bm25.b * doc_len / bm25.avgdl
            )
        )

        contribution = 0.0

        if freq > 0 and denominator != 0:
            contribution = (
                effective_idf
                * freq
                * (bm25.k1 + 1)
                / denominator
            )

        total += contribution

        if freq > 0:
            print(
                f"{token:12} "
                f"freq={freq:<3} "
                f"idf={effective_idf:>9.4f} "
                f"score={contribution:>9.4f}"
            )

    print("CALCULATED TOTAL:", round(total, 6))
