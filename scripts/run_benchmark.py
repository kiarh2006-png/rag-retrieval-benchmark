import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.dense import DenseRetriever
from src.retrieval.lexical import BM25Retriever
from src.retrieval.hybrid import rrf_fuse
from src.evaluation.metrics import evaluate_query, aggregate_results
from src.evaluation.reporting import summarize_by_category, build_failure_candidates


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(config):
    questions = load_json(ROOT / config.get("ground_truth", "data/eval/ground_truth.json"))
    chunks = load_jsonl(ROOT / config.get("corpus", "data/corpus/chunks.jsonl"))
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    for q in questions:
        if q["answerable"] and any(cid not in chunk_lookup for cid in q["relevant_chunk_ids"]):
            raise ValueError(f"Unknown relevant chunk in {q['id']}")

    top_k = int(config.get("top_k", 5))
    candidate_k = int(config.get("candidate_k", 100))
    rrf_k = int(config.get("rrf_k", 60))
    dense_weight = float(config.get("dense_weight", 1.0))
    bm25_weight = float(config.get("bm25_weight", 1.0))
    if top_k < 1 or candidate_k < top_k or rrf_k < 1:
        raise ValueError("Require top_k>=1, candidate_k>=top_k, rrf_k>=1")

    print(f"Loading retrievers | top_k={top_k} candidate_k={candidate_k} rrf_k={rrf_k}")
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    methods = {"Dense": dense, "BM25": bm25, "Hybrid_RRF": None}
    output = {
        "metadata": {
            "experiment_name": config.get("experiment_name", "baseline"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_chunks": len(chunks),
            "questions": len(questions),
            "settings": {"top_k": top_k, "candidate_k": candidate_k, "rrf_k": rrf_k,
                         "dense_model": "bge-m3", "embedding_source": "frozen dense_embeddings.npy",
                         "lexical": "BM25Okapi", "hybrid": "weighted RRF",
                         "dense_weight": dense_weight, "bm25_weight": bm25_weight},
        }
    }

    print("\nPrecomputing candidate lists (one Dense embedding call per question)...")
    candidate_cache = {}
    for i, q in enumerate(questions, start=1):
        print(f"\rCandidates: {i}/{len(questions)} {q['id']}", end="", flush=True)
        candidate_cache[q["id"]] = (
            dense.search(q["question"], top_k=candidate_k),
            bm25.search(q["question"], top_k=candidate_k),
        )
    print()

    for name, retriever in methods.items():
        print(f"\nRunning {name}")
        rows = []
        for i, q in enumerate(questions, start=1):
            print(f"\r{name}: {i}/{len(questions)} {q['id']}", end="", flush=True)
            dense_candidates, lexical_candidates = candidate_cache[q["id"]]
            if name == "Dense":
                found = dense_candidates[:top_k]
            elif name == "BM25":
                found = lexical_candidates[:top_k]
            else:
                found = rrf_fuse(
                    [dense_candidates, lexical_candidates],
                    rrf_k=rrf_k,
                    weights=[dense_weight, bm25_weight],
                    top_k=top_k,
                )
            ids = [x["chunk_id"] for x in found]
            metrics = evaluate_query(ids, q["relevant_chunk_ids"], q["answerable"])
            rows.append({
                "question_id": q["id"], "category": q["category"],
                "answerable": q["answerable"], "retrieved_chunk_ids": ids,
                "metrics": metrics,
            })
        print()
        output[name] = {
            "summary": aggregate_results([r["metrics"] for r in rows]),
            "by_category": summarize_by_category(rows),
            "queries": rows,
        }

    questions_by_id = {q["id"]: q for q in questions}
    output["failure_candidates"] = build_failure_candidates(
        {k: v for k, v in output.items() if k in methods},
        questions_by_id, chunk_lookup
    )

    results_dir = ROOT / config.get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    name = config.get("experiment_name", "baseline")
    out_path = results_dir / f"{name}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    for method in methods:
        print(method, json.dumps(output[method]["summary"], ensure_ascii=False))
    print("Failure candidates:", len(output["failure_candidates"]))
    return output


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    config_path = ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    run(config)


if __name__ == "__main__":
    main()
