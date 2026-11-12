
import json
from pathlib import Path
from collections import Counter

import numpy as np


ROOT = Path(__file__).resolve().parent

CORPUS_PATH = ROOT / "data" / "corpus" / "chunks.jsonl"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "dense_embeddings.npy"
MANIFEST_PATH = ROOT / "data" / "metadata" / "manifest.json"


def load_chunks():
    with open(CORPUS_PATH, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    # Load corpus
    chunks = load_chunks()

    # Load embeddings
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")

    # Load manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    # Document statistics
    document_ids = [chunk["metadata"]["document_id"] for chunk in chunks]

    document_counts = Counter(document_ids)

    # Chunk length statistics
    chunk_lengths = [
        len(chunk["text"])
        for chunk in chunks
    ]

    print("\n[1] CORPUS STATISTICS")

    print("Total chunks:", len(chunks))
    print("Unique documents:", len(document_counts))

    print("\n[2] DOCUMENT DISTRIBUTION")

    for document_id, count in sorted(document_counts.items()):
        print(f"{document_id}: {count} chunks")

    print("\n[3] CHUNK LENGTH STATISTICS")

    print("Minimum:", min(chunk_lengths))
    print("Maximum:", max(chunk_lengths))
    print("Average:", round(np.mean(chunk_lengths), 2))
    print("Median:", round(np.median(chunk_lengths), 2))

    print("\n[4] EMBEDDING STATISTICS")

    print("Shape:", embeddings.shape)
    print("Dtype:", embeddings.dtype)

    print(
        "File size (MB):",
        round(EMBEDDINGS_PATH.stat().st_size / (1024 ** 2), 2)
    )

    print("\n[5] LENGTH DISTRIBUTION")

    bins = [0, 100, 200, 300, 400, 500, 600, float("inf")]

    for lower, upper in zip(bins[:-1], bins[1:]):
        count = sum(
            lower <= length < upper
            for length in chunk_lengths
        )

        print(f"{lower}-{upper}: {count}")

    print("\n[6] MANIFEST")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("STATISTICS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
