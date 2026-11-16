
import json
from pathlib import Path
from collections import defaultdict, Counter

root = Path(".")
data = json.loads(
    (root / "results" / "baseline.json").read_text(encoding="utf-8")
)

failures = data.get("failure_candidates", [])
grouped = defaultdict(list)

for item in failures:
    grouped[item["question_id"]].append(item)

print("\n" + "=" * 65)
print("FAILURE DIAGNOSTIC")
print("=" * 65)

print("Total failure records:", len(failures))
print("Unique failed questions:", len(grouped))

print("\nFailures by retrieval method:")
for method, count in Counter(
    x["retrieval_method"] for x in failures
).items():
    print(f"  {method}: {count}")

print("\nFailures by category:")
for category, count in Counter(
    x["category"] for x in failures
).items():
    print(f"  {category}: {count}")

print("\n" + "=" * 65)
print("QUESTION-LEVEL FAILURE DETAILS")
print("=" * 65)

for qid, records in sorted(grouped.items())[:20]:
    first = records[0]
    print(f"\n{qid} | {first['category']}")
    print("Question:", first["question"])
    print("Expected:", first["expected_relevant_chunks"])

    for record in records:
        method = record["retrieval_method"]
        retrieved = [
            x["chunk_id"] for x in record["retrieved_chunks"]
        ]
        print(f"  {method}: {retrieved}")

print("\n" + "=" * 65)
print("CROSS-METHOD FAILURE QUESTIONS")
print("=" * 65)

for qid, records in sorted(grouped.items()):
    methods = {x["retrieval_method"] for x in records}
    if {"Dense", "BM25", "Hybrid_RRF"}.issubset(methods):
        print(qid, "failed in all three methods")

print("\n" + "=" * 65)
print("EXPERIMENT METADATA")
print("=" * 65)

for filename in [
    "baseline.json",
    "rrf_k_20.json",
    "topk_10.json",
    "weighted_rrf_dense2_bm251.json",
]:
    path = root / "results" / filename
    if not path.exists():
        continue

    result = json.loads(path.read_text(encoding="utf-8"))
    print("\n", filename)
    print(json.dumps(result.get("metadata", {}), ensure_ascii=False, indent=2))

print("\n" + "=" * 65)
print("FINAL REPORT FILE CHECK")
print("=" * 65)

report = root / "data" / "eval" / "final_report.md"
print("Exists:", report.exists())

if report.exists():
    print("Size:", report.stat().st_size, "bytes")
    print("First 20 lines:")
    lines = report.read_text(encoding="utf-8").splitlines()
    for line in lines[:20]:
        print(line)

print("\nDIAGNOSTIC COMPLETE")
