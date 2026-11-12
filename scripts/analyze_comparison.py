
import json
from pathlib import Path

path = Path("data/eval/weighted_comparison.json")

with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

methods = [
    "Dense",
    "BM25",
    "Hybrid_Standard",
    "Hybrid_Dense_Weighted",
]

print("=" * 85)
print("QUESTION-BY-QUESTION COMPARISON")
print("=" * 85)

for i in range(len(data["Dense"]["queries"])):

    rows = {}

    for method in methods:
        item = data[method]["queries"][i]
        metrics = item["metrics"]

        rows[method] = metrics["mrr"]

    question_id = data["Dense"]["queries"][i]["question_id"]

    print(f"\n{question_id}")

    for method, score in rows.items():
        print(f"{method:25} MRR: {score}")

    if rows["Hybrid_Standard"] < rows["Dense"]:
        print("RESULT: Hybrid is WORSE than Dense")

    elif rows["Hybrid_Standard"] > rows["Dense"]:
        print("RESULT: Hybrid is BETTER than Dense")

    else:
        print("RESULT: EQUAL")
