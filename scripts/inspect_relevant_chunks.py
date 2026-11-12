
import json

path = "data/corpus/chunks.jsonl"

target_ids = {
    "taazirat-0130",
    "taazirat-0131",
    "taazirat-0132",
    "taazirat-0133",
    "taazirat-0136",
    "taazirat-0141",
    "civil_law-0083",
}

with open(path, "r", encoding="utf-8") as file:
    for line in file:
        chunk = json.loads(line)

        if chunk["chunk_id"] in target_ids:
            print("\n" + "=" * 80)
            print("CHUNK:", chunk["chunk_id"])
            print("=" * 80)
            print(chunk["text"])
