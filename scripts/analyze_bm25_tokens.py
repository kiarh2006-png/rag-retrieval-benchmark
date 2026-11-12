
from collections import Counter

from src.retrieval.lexical import (
    BM25Retriever,
    tokenize
)

retriever = BM25Retriever()

query = (
    "طبق ماده ۱۵۴ قانون مجازات تعزیرات، "
    "یکی از شرایط پنج‌گانه سرقت چیست؟"
)

target_ids = [
    "taazirat-0130",
    "taazirat-0131",
    "taazirat-0132",
    "taazirat-0133",
    "taazirat-0141",
]

query_tokens = tokenize(query)

print("QUERY TOKENS:")
print(query_tokens)

print("\n" + "=" * 80)

for chunk in retriever.chunks:

    if chunk["chunk_id"] not in target_ids:
        continue

    chunk_tokens = tokenize(chunk["text"])
    counts = Counter(chunk_tokens)

    common = [
        (token, counts[token])
        for token in query_tokens
        if token in counts
    ]

    print("\nCHUNK:", chunk["chunk_id"])
    print("COMMON TOKENS:", common)
    print("TOTAL MATCHES:", sum(count for _, count in common))
