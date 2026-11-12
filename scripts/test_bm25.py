
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.lexical import BM25Retriever

retriever = BM25Retriever()

print("TOTAL CHUNKS:", len(retriever.chunks))

query = "طبق ماده ۱ قانون مالیات‌های مستقیم، اشخاص حقیقی ایرانی مقیم ایران نسبت به چه درآمدهایی مشمول پرداخت مالیات هستند؟"

results = retriever.search(query, top_k=5)

print("\nBM25 TOP 5 RESULTS")
print("=" * 70)

for item in results:
    print(f"\nRank: {item['rank']}")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Score: {item['score']:.4f}")
    print(f"Text: {item['text'][:400]}")
    print("-" * 70)
