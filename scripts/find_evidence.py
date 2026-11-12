
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "corpus" / "chunks.jsonl"


def normalize(text):
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"[\u064b-\u065f]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split()).lower()


def load_chunks():
    chunks = []

    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def search_chunks(chunks, terms, document=None, limit=10):
    terms = normalize(terms).split()
    results = []

    for chunk in chunks:
        metadata = chunk["metadata"]

        if document and metadata["document_id"] != document:
            continue

        text = normalize(chunk["text"])

        score = sum(
            1 for term in terms if term in text
        )

        if score > 0:
            results.append((score, chunk))

    results.sort(
        key=lambda item: (-item[0], item[1]["chunk_id"])
    )

    return results[:limit]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--terms",
        required=True,
        help="Persian search terms"
    )

    parser.add_argument(
        "--document",
        default=None,
        help="Optional document ID"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10
    )

    args = parser.parse_args()

    chunks = load_chunks()

    results = search_chunks(
        chunks,
        args.terms,
        args.document,
        args.limit
    )

    print("=" * 70)
    print("EVIDENCE SEARCH")
    print("=" * 70)

    print("Query:", args.terms)
    print("Results:", len(results))

    for score, chunk in results:
        metadata = chunk["metadata"]

        print("\n" + "-" * 70)
        print("Chunk ID:", chunk["chunk_id"])
        print("Document:", metadata["document_id"])
        print("Title:", metadata["title"])
        print("Page:", metadata.get("page"))
        print("Score:", score)
        print("\nTEXT:")
        print(chunk["text"][:1500])


if __name__ == "__main__":
    main()
