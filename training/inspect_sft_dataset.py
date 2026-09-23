import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
dataset_dir = project_root / "data" / "llm_finetuning_v2"

files = [
    dataset_dir / "train.jsonl",
    dataset_dir / "validation.jsonl",
    dataset_dir / "test.jsonl"
]

for file_path in files:
    print("=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    with open(file_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Examples: {len(records):,}")
    print()

    for index, record in enumerate(records[:3], start=1):
        print(f"EXAMPLE {index}")
        print("-" * 80)

        print(f"Source: {record.get('source')}")
        print(f"Selection type: {record.get('selection_type')}")

        for message in record["messages"]:
            print(f"\n{message['role'].upper()}:")
            print(message["content"][:1000])

        print()

    print()