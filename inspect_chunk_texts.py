
import json
from pathlib import Path

results = json.loads(
    Path("results/baseline.json").read_text(encoding="utf-8")
)

chunks = {}

with open("data/corpus/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        chunk_id = item.get("chunk_id") or item.get("id")
        if chunk_id:
            chunks[chunk_id] = item

selected = [
    "q001", "q002", "q004", "q008",
    "q015", "q018", "q053", "q060",
    "q061", "q062", "q063", "q064",
    "q066", "q067", "q083", "q085"
]

failures = results["failure_candidates"]
by_question = {}

for f in failures:
    by_question.setdefault(f["question_id"], {})[
        f["retrieval_method"]
    ] = f

output = []

def get_text(item):
    for key in ["text", "content", "chunk_text", "document"]:
        value = item.get(key)
        if isinstance(value, str):
            return value
    return str(item)[:500]

for qid in selected:
    methods = by_question.get(qid, {})
    if not methods:
        continue

    sample = next(iter(methods.values()))

    output.append("\n" + "=" * 75)
    output.append(f"QUESTION: {qid}")
    output.append(sample["question"])
    output.append("EXPECTED: " + str(sample["expected_relevant_chunks"]))

    ids = []

    for expected in sample["expected_relevant_chunks"]:
        if expected not in ids:
            ids.append(expected)

    for method in ["Dense", "BM25", "Hybrid_RRF"]:
        if method not in methods:
            continue
        for hit in methods[method]["retrieved_chunks"][:3]:
            cid = hit["chunk_id"]
            if cid not in ids:
                ids.append(cid)

    for cid in ids:
        item = chunks.get(cid)
        output.append("\n--- CHUNK: " + cid + " ---")

        if item is None:
            output.append("CHUNK NOT FOUND")
        else:
            output.append(get_text(item)[:550])

    for method in ["Dense", "BM25", "Hybrid_RRF"]:
        if method in methods:
            ids_found = [
                x["chunk_id"]
                for x in methods[method]["retrieved_chunks"][:5]
            ]
            output.append(f"\n{method}: {ids_found}")

Path("chunk_inspection.txt").write_text(
    "\n".join(output),
    encoding="utf-8"
)

print("INSPECTION COMPLETE")
print("Questions inspected:", len(selected))
print("Chunks loaded:", len(chunks))
print("Output: chunk_inspection.txt")
print("Open the output with Notepad.")
