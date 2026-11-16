import json
from pathlib import Path

root = Path.cwd()

def read(path):
    return json.loads((root / path).read_text(encoding="utf-8"))

print("=" * 65)
print("FINAL PROJECT AUDIT")
print("=" * 65)

# Ground truth
gt = read("data/eval/ground_truth.json")
print("\nQUESTIONS:", len(gt))
print("ANSWERABLE:", sum(q["answerable"] for q in gt))
print("UNANSWERABLE:", sum(not q["answerable"] for q in gt))

# Corpus
chunks = {}
with open(root / "data/corpus/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            chunks[item["chunk_id"]] = item

missing = [
    cid
    for q in gt
    for cid in q.get("relevant_chunk_ids", [])
    if cid not in chunks
]
print("CORPUS CHUNKS:", len(chunks))
print("MISSING REFERENCES:", len(missing))

# q004
q = next(q for q in gt if q["id"] == "q004")
print("\nQ004 EXPECTED:", q["relevant_chunk_ids"])
for cid in q["relevant_chunk_ids"]:
    print("TEXT:", chunks[cid]["text"][:500])

# Experiment summaries
print("\n" + "=" * 65)
print("EXPERIMENT RESULTS")
print("=" * 65)

for path in sorted((root / "results").glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not all(m in data for m in ("Dense", "BM25", "Hybrid_RRF")):
        continue

    print("\n", path.name)
    for method in ("Dense", "BM25", "Hybrid_RRF"):
        summary = data[method].get("summary", {})
        print(
            method,
            "hit@1 =", summary.get("hit@1"),
            "| hit@5 =", summary.get("hit@5"),
            "| recall@5 =", summary.get("recall@5"),
            "| MRR =", summary.get("mrr"),
            "| NDCG@5 =", summary.get("ndcg@5")
        )

# Failure summary
baseline = read("results/baseline.json")
failures = baseline.get("failure_candidates", [])
print("\nFAILURE RECORDS:", len(failures))

# Test status
print("\n" + "=" * 65)
print("AUDIT COMPLETE")
print("=" * 65)
