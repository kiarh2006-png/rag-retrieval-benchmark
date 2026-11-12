
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "data" / "corpus" / "chunks.jsonl"
OUTPUT_PATH = ROOT / "data" / "eval" / "corpus_inventory.md"


def load_chunks():
    chunks = []

    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def main():
    chunks = load_chunks()

    documents = defaultdict(list)

    for chunk in chunks:
        metadata = chunk["metadata"]
        document_id = metadata["document_id"]

        documents[document_id].append(chunk)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:

        file.write("# Corpus Inventory\n\n")
        file.write(f"Total chunks: {len(chunks)}\n")
        file.write(f"Unique documents: {len(documents)}\n\n")

        for document_id, items in sorted(documents.items()):

            first = items[0]
            metadata = first["metadata"]

            file.write("\n---\n\n")
            file.write(f"## {document_id}\n\n")

            file.write(
                f"- Title: {metadata.get('title')}\n"
            )
            file.write(
                f"- Type: {metadata.get('doc_type')}\n"
            )
            file.write(
                f"- Filename: {metadata.get('filename')}\n"
            )
            file.write(
                f"- Total chunks: {len(items)}\n\n"
            )

            # Select representative chunks
            indexes = sorted(set([
                0,
                len(items) // 2,
                len(items) - 1
            ]))

            for index in indexes:

                chunk = items[index]

                file.write(
                    f"### Chunk: {chunk['chunk_id']}\n\n"
                )

                file.write(
                    chunk["text"][:1200] + "\n\n"
                )

    print("CORPUS INVENTORY COMPLETED")
    print("Documents:", len(documents))
    print("Chunks:", len(chunks))
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
