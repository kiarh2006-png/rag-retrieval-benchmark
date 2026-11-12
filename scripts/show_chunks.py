import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "corpus" / "chunks.jsonl"

TARGET_IDS = {
    "taazirat-0130",
    "taazirat-0131",
    "taazirat-0132",
    "penal_code-0102",
    "civil_law-0080",
    "civil_law-0081",
}


def main():
    found = set()

    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunk = json.loads(line)

            if chunk["chunk_id"] not in TARGET_IDS:
                continue

            metadata = chunk["metadata"]
            found.add(chunk["chunk_id"])

            print("\n" + "=" * 80)
            print("CHUNK ID:", chunk["chunk_id"])
            print("DOCUMENT:", metadata["document_id"])
            print("TITLE:", metadata["title"])
            print("PAGE:", metadata.get("page"))
            print("-" * 80)
            print(chunk["text"])

    print("\n" + "=" * 80)
    print("FOUND:", len(found), "OF", len(TARGET_IDS))
    print("MISSING:", sorted(TARGET_IDS - found))


if __name__ == "__main__":
    main()