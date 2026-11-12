

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_PATH = ROOT / "data" / "eval" / "ground_truth.json"
CORPUS_PATH = ROOT / "data" / "corpus" / "chunks.jsonl"

VALID_CATEGORIES = {
    "simple",
    "paraphrase",
    "multi_hop",
    "legal_terminology",
    "similar_documents",
    "unanswerable",
}

REQUIRED_FIELDS = {
    "id",
    "question",
    "category",
    "expected_documents",
    "relevant_chunk_ids",
    "answerable",
}


def load_corpus():
    chunks = {}

    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunk = json.loads(line)
            chunks[chunk["chunk_id"]] = chunk["metadata"]["document_id"]

    return chunks


def validate():
    errors = []

    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    corpus = load_corpus()

    seen_ids = set()

    for index, item in enumerate(questions, start=1):
        prefix = f"Question #{index}"

        if not isinstance(item, dict):
            errors.append(f"{prefix}: must be a JSON object")
            continue

        missing = REQUIRED_FIELDS - set(item.keys())

        if missing:
            errors.append(f"{prefix}: missing fields {sorted(missing)}")
            continue

        question_id = item["id"]

        if not isinstance(question_id, str) or not question_id.strip():
            errors.append(f"{prefix}: invalid question ID")
        elif question_id in seen_ids:
            errors.append(f"{prefix}: duplicate ID {question_id}")
        else:
            seen_ids.add(question_id)

        if not isinstance(item["question"], str) or not item["question"].strip():
            errors.append(f"{prefix}: empty question")

        if item["category"] not in VALID_CATEGORIES:
            errors.append(
                f"{prefix}: invalid category {item['category']}"
            )

        if not isinstance(item["answerable"], bool):
            errors.append(f"{prefix}: answerable must be boolean")
            continue

        docs = item["expected_documents"]
        chunk_ids = item["relevant_chunk_ids"]

        if not isinstance(docs, list) or not all(
            isinstance(doc, str) for doc in docs
        ):
            errors.append(f"{prefix}: expected_documents must be a list of strings")
            continue

        if not isinstance(chunk_ids, list) or not all(
            isinstance(cid, str) for cid in chunk_ids
        ):
            errors.append(f"{prefix}: relevant_chunk_ids must be a list of strings")
            continue

        if len(docs) != len(set(docs)):
            errors.append(f"{prefix}: duplicate expected documents")

        if len(chunk_ids) != len(set(chunk_ids)):
            errors.append(f"{prefix}: duplicate relevant chunk IDs")

        if item["answerable"]:
            if not docs:
                errors.append(f"{prefix}: answerable question has no expected documents")

            if not chunk_ids:
                errors.append(f"{prefix}: answerable question has no relevant chunks")

        else:
            if docs or chunk_ids:
                errors.append(
                    f"{prefix}: unanswerable question must have empty document and chunk lists"
                )

        found_documents = set()

        for chunk_id in chunk_ids:
            if chunk_id not in corpus:
                errors.append(f"{prefix}: unknown chunk ID {chunk_id}")
                continue

            document_id = corpus[chunk_id]
            found_documents.add(document_id)

            if document_id not in docs:
                errors.append(
                    f"{prefix}: chunk {chunk_id} belongs to {document_id}, "
                    "which is not in expected_documents"
                )

        if item["answerable"] and found_documents != set(docs):
            errors.append(
                f"{prefix}: every expected document must have at least one relevant chunk"
            )

    print("=" * 60)
    print("GROUND TRUTH VALIDATION")
    print("=" * 60)
    print("Questions:", len(questions))
    print("Unique IDs:", len(seen_ids))
    print("Corpus chunks:", len(corpus))

    if errors:
        print("\nVALIDATION FAILED")
        for error in errors:
            print("-", error)
        raise SystemExit(1)

    print("\nVALIDATION PASSED")
    print("All structural and reference checks passed.")


if __name__ == "__main__":
    validate()
