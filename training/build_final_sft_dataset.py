import json
import random
from collections import Counter, defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

pku_file = project_root / "data" / "llm_finetuning_v2" / "clean_candidates.jsonl"
project_file = project_root / "data" / "project_sft" / "safety_response_examples.jsonl"

output_dir = project_root / "data" / "llm_finetuning_final"

train_file = output_dir / "train.jsonl"
validation_file = output_dir / "validation.jsonl"
test_file = output_dir / "test.jsonl"

output_dir.mkdir(parents=True, exist_ok=True)

random.seed(42)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


pku_records = load_jsonl(pku_file)
project_records = load_jsonl(project_file)

manual_review_prompt = "I really want to learn how to hack someone's bank account."

pku_records = [
    record
    for record in pku_records
    if record["messages"][0]["content"].strip() != manual_review_prompt
]

all_records = pku_records + project_records

seen_pairs = set()
unique_records = []

for record in all_records:
    prompt = record["messages"][0]["content"].strip()
    response = record["messages"][1]["content"].strip()

    key = (prompt, response)

    if key in seen_pairs:
        continue

    seen_pairs.add(key)
    unique_records.append(record)

groups = defaultdict(list)

for record in unique_records:
    prompt = record["messages"][0]["content"].strip()
    groups[prompt].append(record)

prompt_groups = list(groups.items())
random.shuffle(prompt_groups)

total_records = len(unique_records)

target_train = int(total_records * 0.80)
target_validation = int(total_records * 0.10)

train_records = []
validation_records = []
test_records = []

for prompt, records in prompt_groups:
    if len(train_records) < target_train:
        train_records.extend(records)
    elif len(validation_records) < target_validation:
        validation_records.extend(records)
    else:
        test_records.extend(records)

random.shuffle(train_records)
random.shuffle(validation_records)
random.shuffle(test_records)

write_jsonl(train_file, train_records)
write_jsonl(validation_file, validation_records)
write_jsonl(test_file, test_records)

train_prompts = {
    record["messages"][0]["content"].strip()
    for record in train_records
}

validation_prompts = {
    record["messages"][0]["content"].strip()
    for record in validation_records
}

test_prompts = {
    record["messages"][0]["content"].strip()
    for record in test_records
}

train_validation_overlap = train_prompts & validation_prompts
train_test_overlap = train_prompts & test_prompts
validation_test_overlap = validation_prompts & test_prompts

source_counts = Counter()
category_counts = Counter()

for record in unique_records:
    source_counts[record.get("source", "unknown")] += 1
    category_counts[
        record.get(
            "category",
            record.get("selection_type", "unknown")
        )
    ] += 1

train_source_counts = Counter(
    record.get("source", "unknown")
    for record in train_records
)

validation_source_counts = Counter(
    record.get("source", "unknown")
    for record in validation_records
)

test_source_counts = Counter(
    record.get("source", "unknown")
    for record in test_records
)

print("=" * 70)
print("FINAL LLAMA SFT DATASET")
print("=" * 70)

print(f"PKU records after QC:         {len(pku_records):,}")
print(f"Project records:               {len(project_records):,}")
print(f"Unique records:                {len(unique_records):,}")
print(f"Unique prompts:                {len(prompt_groups):,}")
print()

print("FINAL SPLIT")
print("-" * 70)
print(f"Training:                      {len(train_records):,}")
print(f"Validation:                    {len(validation_records):,}")
print(f"Test:                          {len(test_records):,}")
print()

print("SOURCE DISTRIBUTION")
print("-" * 70)
print(
    f"Train PKU:                     "
    f"{train_source_counts.get('PKU-SafeRLHF-30K', 0):,}"
)
print(
    f"Train project:                 "
    f"{train_source_counts.get('project_safety_response', 0):,}"
)
print(
    f"Validation PKU:                "
    f"{validation_source_counts.get('PKU-SafeRLHF-30K', 0):,}"
)
print(
    f"Validation project:            "
    f"{validation_source_counts.get('project_safety_response', 0):,}"
)
print(
    f"Test PKU:                      "
    f"{test_source_counts.get('PKU-SafeRLHF-30K', 0):,}"
)
print(
    f"Test project:                  "
    f"{test_source_counts.get('project_safety_response', 0):,}"
)
print()

print("PROMPT LEAKAGE CHECK")
print("-" * 70)
print(f"Train/Validation overlap:      {len(train_validation_overlap)}")
print(f"Train/Test overlap:            {len(train_test_overlap)}")
print(f"Validation/Test overlap:       {len(validation_test_overlap)}")
print()

print("SOURCE TOTALS")
print("-" * 70)

for source, count in sorted(source_counts.items()):
    print(f"{source:<35} {count:,}")

print()

print("CATEGORY DISTRIBUTION")
print("-" * 70)

for category, count in sorted(category_counts.items()):
    print(f"{category:<35} {count:,}")

print()

print("OUTPUT FILES")
print("-" * 70)
print(train_file)
print(validation_file)
print(test_file)

print("=" * 70)