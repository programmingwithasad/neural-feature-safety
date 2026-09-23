import json
import lzma
from pathlib import Path
from collections import Counter

project_root = Path(__file__).resolve().parents[1]
dataset_file = project_root / "data" / "raw" / "pku_saferlhf" / "train.jsonl.xz"

total = 0
both_safe = 0
only_response_0_safe = 0
only_response_1_safe = 0
both_unsafe = 0
better_response_safe = 0
safer_response_safe = 0

prompt_lengths = []
response_lengths = []

with lzma.open(dataset_file, "rt", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        total += 1

        safe_0 = bool(record["is_response_0_safe"])
        safe_1 = bool(record["is_response_1_safe"])

        response_0 = record["response_0"]
        response_1 = record["response_1"]

        prompt_lengths.append(len(record["prompt"]))
        response_lengths.append(len(response_0))
        response_lengths.append(len(response_1))

        if safe_0 and safe_1:
            both_safe += 1
        elif safe_0 and not safe_1:
            only_response_0_safe += 1
        elif not safe_0 and safe_1:
            only_response_1_safe += 1
        else:
            both_unsafe += 1

        better_id = record["better_response_id"]
        safer_id = record["safer_response_id"]

        better_safe = safe_0 if better_id == 0 else safe_1
        safer_safe = safe_0 if safer_id == 0 else safe_1

        if better_safe:
            better_response_safe += 1

        if safer_safe:
            safer_response_safe += 1

print("=" * 70)
print("PKU-SafeRLHF-30K TRAINING DATASET ANALYSIS")
print("=" * 70)

print(f"Total examples: {total:,}")
print()

print("SAFETY DISTRIBUTION")
print("-" * 70)
print(f"Both responses safe:              {both_safe:,}")
print(f"Only response 0 safe:             {only_response_0_safe:,}")
print(f"Only response 1 safe:             {only_response_1_safe:,}")
print(f"Both responses unsafe:             {both_unsafe:,}")
print()

print("SELECTED RESPONSE SAFETY")
print("-" * 70)
print(f"Better response is safe:          {better_response_safe:,}")
print(f"Safer response is actually safe:  {safer_response_safe:,}")
print()

print("PERCENTAGES")
print("-" * 70)

if total:
    print(f"Both safe:                         {both_safe / total * 100:.2f}%")
    print(f"Only response 0 safe:             {only_response_0_safe / total * 100:.2f}%")
    print(f"Only response 1 safe:             {only_response_1_safe / total * 100:.2f}%")
    print(f"Both unsafe:                       {both_unsafe / total * 100:.2f}%")
    print(f"Better response safe:             {better_response_safe / total * 100:.2f}%")
    print(f"Safer response actually safe:     {safer_response_safe / total * 100:.2f}%")

print()

print("LENGTH STATISTICS")
print("-" * 70)

if prompt_lengths:
    print(f"Average prompt length:             {sum(prompt_lengths) / len(prompt_lengths):.1f} characters")

if response_lengths:
    print(f"Average response length:           {sum(response_lengths) / len(response_lengths):.1f} characters")

print("=" * 70)