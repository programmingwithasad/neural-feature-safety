import csv
import json
import re
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

input_file = project_root / "data" / "llm_finetuning_v2" / "train.jsonl"
output_file = project_root / "data" / "processed" / "pku_flagged_examples.csv"

output_file.parent.mkdir(parents=True, exist_ok=True)

patterns = {
    "hate_slur": [
        r"\bnigger\b",
        r"\bnigga\b",
        r"\bchink\b",
        r"\bspic\b",
        r"\bkike\b",
        r"\bwetback\b"
    ],
    "weapon_instruction": [
        r"\bhow to make a bomb\b",
        r"\bhow to build a bomb\b",
        r"\bmake an explosive\b",
        r"\bbuild an explosive\b",
        r"\bmake a gun\b",
        r"\bbuild a gun\b"
    ],
    "privacy": [
        r"\bhome address\b",
        r"\bresidential address\b",
        r"\bphone number\b",
        r"\bsocial security\b",
        r"\bcredit card number\b"
    ],
    "vandalism": [
        r"\bgraffiti\b",
        r"\bdestroy.*property\b",
        r"\bdamage.*property\b",
        r"\bvandal"
    ],
    "violence": [
        r"\bkill\b",
        r"\bmurder\b",
        r"\bshoot\b",
        r"\bstab\b",
        r"\bpoison\b"
    ],
    "sexual": [
        r"\bsex\b",
        r"\bsexual\b",
        r"\bintercourse\b",
        r"\bgenitals?\b",
        r"\bpussy\b",
        r"\bbitches\b"
    ],
    "fraud_manipulation": [
        r"\bscam\b",
        r"\bfraud\b",
        r"\bphishing\b",
        r"\bsteal\b",
        r"\bmanipulate\b"
    ]
}

compiled_patterns = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in values]
    for category, values in patterns.items()
}

rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for index, line in enumerate(f):
        if not line.strip():
            continue

        record = json.loads(line)

        prompt = record["messages"][0]["content"]
        response = record["messages"][1]["content"]

        categories = []

        for category, regexes in compiled_patterns.items():
            if any(regex.search(response) for regex in regexes):
                categories.append(category)

        if categories:
            rows.append(
                {
                    "index": index,
                    "categories": "|".join(categories),
                    "prompt": prompt,
                    "response": response,
                    "selection_type": record.get("selection_type", "")
                }
            )

with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "index",
            "categories",
            "prompt",
            "response",
            "selection_type"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print("=" * 70)
print("FLAGGED SFT EXAMPLES EXPORTED")
print("=" * 70)
print(f"Flagged examples: {len(rows):,}")
print(f"Output: {output_file}")
print("=" * 70)