import json
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]


def test_ground_truth_structure_and_references():
    qs = json.loads((ROOT / "data/eval/ground_truth.json").read_text(encoding="utf-8"))
    chunks = [json.loads(x) for x in (ROOT / "data/corpus/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    doc_by_id = {c["chunk_id"]: c["metadata"]["document_id"] for c in chunks}
    assert len(qs) >= 100
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert {"id","question","category","expected_documents","relevant_chunk_ids","answerable"} <= q.keys()
        assert q["question"].strip()
        if q["answerable"]:
            assert q["relevant_chunk_ids"] and q["expected_documents"]
            assert set(q["relevant_chunk_ids"]) <= doc_by_id.keys()
            assert {doc_by_id[c] for c in q["relevant_chunk_ids"]} == set(q["expected_documents"])
        else:
            assert q["relevant_chunk_ids"] == []
            assert q["expected_documents"] == []


def test_unanswerable_and_category_coverage():
    qs = json.loads((ROOT / "data/eval/ground_truth.json").read_text(encoding="utf-8"))
    categories = {q["category"] for q in qs}
    assert {"simple","paraphrase","multi_hop","legal_terminology","similar_documents","unanswerable"} <= categories
    assert sum(not q["answerable"] for q in qs) >= 1
