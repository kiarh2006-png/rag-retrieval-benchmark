
import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.dense import DenseRetriever

retriever = DenseRetriever()

print("TOTAL CHUNKS:", len(retriever.chunks))
print("EMBEDDING SHAPE:", retriever.embeddings.shape)
print("DIMENSION:", retriever.dimension)

query = "طبق ماده ۱ قانون مالیات‌های مستقیم، اشخاص حقیقی ایرانی مقیم ایران نسبت به چه درآمدهایی مشمول پرداخت مالیات هستند؟"

results = retriever.search(query, top_k=5)

print("\nTOP 5 RESULTS")
print("=" * 70)

for item in results:
    print(f"\nRank: {item['rank']}")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Score: {item['score']:.4f}")
    print(f"Text: {item['text'][:500]}")
    print("-" * 70)
