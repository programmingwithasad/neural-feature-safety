import json
import lzma
import random
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

source_file = project_root / "data" / "raw" / "pku_saferlhf" / "train.jsonl.xz"
output_dir = project_root / "data" / "llm_finetuning_v2"

train_file = output_dir / "train.jsonl"
validation_file = output_dir / "validation.jsonl"
test_file = output_dir / "test.jsonl"

output_dir.mkdir(parents=True, exist_ok=True)

random.seed(42)

records = []
seen_pairs = set()

with lzma.open(source_file, "rt", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        prompt = record["prompt"].strip()
        response_0 = record["response_0"].strip()
        response_1 = record["response_1"].strip()

        safe_0 = bool(record["is_response_0_safe"])
        safe_1 = bool(record["is_response_1_safe"])

        if not prompt or not response_0 or not response_1:
            continue

        selected_response = None
        selection_type = None

        if safe_0 and safe_1:
            better_id = int(record["better_response_id"])

            if better_id == 0:
                selected_response = response_0
            elif better_id == 1:
                selected_response = response_1

            selection_type = "both_safe_better_response"

        elif safe_0 and not safe_1:
            selected_response = response_0
            selection_type = "safe_response_0"

        elif not safe_0 and safe_1:
            selected_response = response_1
            selection_type = "safe_response_1"

        if selected_response is None:
            continue

        if len(prompt) < 5 or len(selected_response) < 10:
            continue

        pair_key = (prompt, selected_response)

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)

        records.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": selected_response
                    }
                ],
                "source": "PKU-SafeRLHF-30K",
                "selection_type": selection_type
            }
        )

groups = defaultdict(list)

for record in records:
    prompt = record["messages"][0]["content"]
    groups[prompt].append(record)

prompt_groups = list(groups.values())
random.shuffle(prompt_groups)

total = len(records)
target_train = int(total * 0.80)
target_validation = int(total * 0.10)

train_records = []
validation_records = []
test_records = []

for group in prompt_groups:
    if len(train_records) < target_train:
        train_records.extend(group)
    elif len(validation_records) < target_validation:
        validation_records.extend(group)
    else:
        test_records.extend(group)

random.shuffle(train_records)
random.shuffle(validation_records)
random.shuffle(test_records)

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

write_jsonl(train_file, train_records)
write_jsonl(validation_file, validation_records)
write_jsonl(test_file, test_records)

selection_counts = defaultdict(int)

for record in records:
    selection_counts[record["selection_type"]] += 1

train_prompts = {
    record["messages"][0]["content"]
    for record in train_records
}

validation_prompts = {
    record["messages"][0]["content"]
    for record in validation_records
}

test_prompts = {
    record["messages"][0]["content"]
    for record in test_records
}

train_validation_overlap = train_prompts & validation_prompts
train_test_overlap = train_prompts & test_prompts
validation_test_overlap = validation_prompts & test_prompts

print("=" * 70)
print("PKU-SafeRLHF-30K SFT DATASET PREPARATION")
print("=" * 70)

print(f"Source examples:              26,874")
print(f"Usable safe examples:         {total:,}")
print(f"Unique prompts:               {len(prompt_groups):,}")
print()

print("SELECTION DISTRIBUTION")
print("-" * 70)

for key, value in sorted(selection_counts.items()):
    print(f"{key:<35} {value:,}")

print()

print("FINAL SPLIT")
print("-" * 70)
print(f"Training examples:            {len(train_records):,}")
print(f"Validation examples:          {len(validation_records):,}")
print(f"Test examples:                {len(test_records):,}")

print()

print("PROMPT LEAKAGE CHECK")
print("-" * 70)
print(f"Train/Validation overlap:     {len(train_validation_overlap)}")
print(f"Train/Test overlap:           {len(train_test_overlap)}")
print(f"Validation/Test overlap:      {len(validation_test_overlap)}")

print()

print("OUTPUT FILES")
print("-" * 70)
print(train_file)
print(validation_file)
print(test_file)

print("=" * 70)