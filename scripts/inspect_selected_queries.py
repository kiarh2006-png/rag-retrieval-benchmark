
import json
from pathlib import Path

path = Path("data/eval/weighted_comparison.json")

with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

selected = ["q001", "q003", "q010", "q015"]

methods = [
    "Dense",
    "BM25",
    "Hybrid_Standard",
    "Hybrid_Dense_Weighted",
]

for question_id in selected:

    print("\n" + "=" * 80)
    print("QUESTION:", question_id)
    print("=" * 80)

    for method in methods:

        item = next(
            x for x in data[method]["queries"]
            if x["question_id"] == question_id
        )

        print("\nMETHOD:", method)
        print("MRR:", item["metrics"]["mrr"])
        print("Retrieved chunks:")

        for rank, chunk_id in enumerate(
            item["retrieved_chunk_ids"], start=1
        ):
            print(f"  {rank}. {chunk_id}")
