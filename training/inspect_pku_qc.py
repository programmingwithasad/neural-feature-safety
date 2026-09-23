import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

review_file = project_root / "data" / "processed" / "pku_manual_review.jsonl"

with open(review_file, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

print("=" * 80)
print("MANUAL REVIEW RECORDS")
print("=" * 80)

print(f"Records requiring review: {len(records)}")
print()

for item in records:
    record = item["record"]

    print(f"INDEX: {item['index']}")
    print(f"REASON: {item['reason']}")
    print(f"CATEGORIES: {item['categories']}")
    print("-" * 80)

    for message in record["messages"]:
        print(f"\n{message['role'].upper()}:")
        print(message["content"])

    print()
    print("=" * 80)