import json
import re
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

input_file = project_root / "data" / "llm_finetuning_v2" / "train.jsonl"
output_file = project_root / "data" / "llm_finetuning_v2" / "clean_candidates.jsonl"
review_file = project_root / "data" / "processed" / "pku_manual_review.jsonl"

output_file.parent.mkdir(parents=True, exist_ok=True)
review_file.parent.mkdir(parents=True, exist_ok=True)

dangerous_patterns = {
    "weapon_action": [
        r"\bhere(?:'s| is) how to make\b.*\b(bomb|explosive)\b",
        r"\bsteps?\s+(to|for)\s+(make|build)\b.*\b(bomb|explosive|weapon)\b",
        r"\bfirst\b.*\bthen\b.*\b(finally|after that)\b.*\b(explosive|bomb)\b"
    ],
    "violent_action": [
        r"\bhere(?:'s| is) how to\b.*\b(kill|murder|stab|shoot|poison)\b",
        r"\bsteps?\s+(to|for)\s+(kill|murder|stab|shoot|poison)\b",
        r"\bfirst\b.*\bthen\b.*\b(kill|murder|stab|shoot|poison)\b"
    ],
    "property_crime": [
        r"\bhere(?:'s| is) how to\b.*\b(steal|rob|burglarize|vandalize)\b",
        r"\bsteps?\s+(to|for)\s+(steal|rob|burglarize|vandalize)\b",
        r"\bhow to\b.*\bget away with\b.*\b(stealing|robbery|vandalism)\b"
    ],
    "privacy_violation": [
        r"\bhere(?:'s| is)\b.*\b(home address|residential address|social security number|credit card number)\b",
        r"\btheir\b.*\b(home address|residential address|social security number|phone number)\b.*\bis\b",
        r"\bfind\b.*\b(private|residential)\b.*\b(address|phone number)\b"
    ],
    "sexual_action": [
        r"\bhere(?:'s| is) how to\b.*\b(sex|sexual|intercourse)\b.*\bwithout\b.*\b(consent|permission)\b",
        r"\bsteps?\s+(to|for)\b.*\b(sexual assault|rape)\b",
        r"\bhow to\b.*\bforce\b.*\bsex\b"
    ],
    "fraud_action": [
        r"\bhere(?:'s| is) how to\b.*\b(fraud|scam|phish)\b",
        r"\bsteps?\s+(to|for)\b.*\b(fraud|scam|phishing)\b",
        r"\bhow to\b.*\bsteal\b.*\b(identity|money|account)\b"
    ]
}

refusal_patterns = [
    r"\bi can't\b",
    r"\bi cannot\b",
    r"\bi'm unable\b",
    r"\bi am unable\b",
    r"\bi won't\b",
    r"\bi will not\b",
    r"\bi wouldn't\b",
    r"\bi would not\b",
    r"\bcan't help\b",
    r"\bcannot help\b",
    r"\bcan't provide\b",
    r"\bcannot provide\b",
    r"\bcan't assist\b",
    r"\bcannot assist\b",
    r"\bnot able to help\b",
    r"\bnot able to provide\b",
    r"\bnot appropriate\b",
    r"\bnot safe\b",
    r"\billegal\b",
    r"\bharmful\b",
    r"\bdangerous\b",
    r"\bprivacy\b"
]

compiled_dangerous = {
    category: [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns]
    for category, patterns in dangerous_patterns.items()
}

compiled_refusals = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in refusal_patterns
]

def contains_refusal(text):
    return any(pattern.search(text) for pattern in compiled_refusals)

def matched_dangerous_categories(text):
    matches = []

    for category, patterns in compiled_dangerous.items():
        if any(pattern.search(text) for pattern in patterns):
            matches.append(category)

    return matches

clean_records = []
review_records = []

stats = Counter()

with open(input_file, "r", encoding="utf-8") as f:
    for index, line in enumerate(f):
        if not line.strip():
            continue

        record = json.loads(line)

        prompt = record["messages"][0]["content"].strip()
        response = record["messages"][1]["content"].strip()

        dangerous_matches = matched_dangerous_categories(response)
        refusal = contains_refusal(response)

        if dangerous_matches:
            if refusal:
                review_records.append(
                    {
                        "index": index,
                        "reason": "dangerous_pattern_with_refusal",
                        "categories": dangerous_matches,
                        "record": record
                    }
                )
                stats["manual_review"] += 1
            else:
                stats["removed_unsafe_actionable"] += 1

        else:
            clean_records.append(record)
            stats["kept"] += 1

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

write_jsonl(output_file, clean_records)
write_jsonl(review_file, review_records)

print("=" * 70)
print("PKU-SafeRLHF RESPONSE QUALITY CONTROL")
print("=" * 70)

print(f"Input records:                 {stats['kept'] + stats['removed_unsafe_actionable'] + stats['manual_review']:,}")
print(f"Kept automatically:            {stats['kept']:,}")
print(f"Removed as actionable unsafe:  {stats['removed_unsafe_actionable']:,}")
print(f"Manual review:                 {stats['manual_review']:,}")

print()
print("OUTPUTS")
print("-" * 70)
print(f"Clean candidates:              {output_file}")
print(f"Manual review:                 {review_file}")

print("=" * 70)