import json
import re
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
dataset_file = project_root / "data" / "llm_finetuning_v2" / "train.jsonl"

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
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns_list]
    for category, patterns_list in patterns.items()
}

counts = Counter()
examples = {}

with open(dataset_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        prompt = record["messages"][0]["content"]
        response = record["messages"][1]["content"]

        matched_categories = []

        for category, regexes in compiled_patterns.items():
            if any(regex.search(response) for regex in regexes):
                matched_categories.append(category)

        if matched_categories:
            for category in matched_categories:
                counts[category] += 1

                if category not in examples:
                    examples[category] = {
                        "prompt": prompt,
                        "response": response
                    }

        else:
            counts["clean_by_pattern"] += 1

print("=" * 80)
print("SFT RESPONSE QUALITY ANALYSIS")
print("=" * 80)

total = sum(counts.values())

for category, count in counts.most_common():
    print(f"{category:<30} {count:,}")

print()
print(f"Total pattern matches: {total:,}")
print()

print("=" * 80)
print("REPRESENTATIVE FLAGGED EXAMPLES")
print("=" * 80)

for category, example in examples.items():
    print()
    print(f"[{category.upper()}]")
    print("-" * 80)
    print("USER:")
    print(example["prompt"][:500])
    print()
    print("ASSISTANT:")
    print(example["response"][:1000])

print()
print("=" * 80)