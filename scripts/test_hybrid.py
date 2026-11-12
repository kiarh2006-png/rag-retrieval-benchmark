
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever

retriever = HybridRetriever(rrf_k=60)

query = "طبق ماده ۱ قانون مالیات‌های مستقیم، اشخاص حقیقی ایرانی مقیم ایران نسبت به چه درآمدهایی مشمول پرداخت مالیات هستند؟"

results = retriever.search(
    query,
    top_k=5,
    candidate_k=100
)

print("HYBRID TOP 5 RESULTS")
print("=" * 70)

for item in results:
    print(f"\nRank: {item['rank']}")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"RRF Score: {item['rrf_score']:.6f}")
    print(f"Text: {item['text'][:400]}")
    print("-" * 70)
