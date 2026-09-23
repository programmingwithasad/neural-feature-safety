import json
import os
import statistics
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.safety_model import SafetyModel


dataset_path = os.path.join(
    project_root,
    "data",
    "processed",
    "refusal_diagnostic_dataset.json"
)

results_path = os.path.join(
    project_root,
    "data",
    "processed",
    "refusal_diagnostic_results.json"
)


with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)


model = SafetyModel()


results = []

for index, sample in enumerate(dataset, start=1):

    result = model.analyze(sample["text"])

    results.append({
        "index": index,
        "category": sample["category"],
        "text": sample["text"],
        "harmful_probability": result["harmful_probability"],
        "unharmful_probability": result["unharmful_probability"],
        "decision": result["decision"],
        "feature_850_activation": result["feature_850_activation"]
    })

categories = {}

for result in results:
    category = result["category"]

    if category not in categories:
        categories[category] = []

    categories[category].append(result)


print("=" * 80)
print("REFUSAL FALSE-POSITIVE DIAGNOSTIC")
print("=" * 80)

summary = {}

for category, category_results in categories.items():

    probabilities = [
        result["harmful_probability"]
        for result in category_results
    ]

    blocked = sum(
        1
        for result in category_results
        if result["decision"] == "BLOCK"
    )

    allowed = len(category_results) - blocked

    mean_probability = statistics.mean(probabilities)
    median_probability = statistics.median(probabilities)

    summary[category] = {
        "count": len(category_results),
        "mean_harmful_probability": mean_probability,
        "median_harmful_probability": median_probability,
        "minimum_harmful_probability": min(probabilities),
        "maximum_harmful_probability": max(probabilities),
        "blocked": blocked,
        "allowed": allowed,
        "block_rate": blocked / len(category_results),
        "allow_rate": allowed / len(category_results)
    }

    print("\n" + "-" * 80)
    print(category)
    print("-" * 80)

    print(f"Samples: {len(category_results)}")
    print(f"Mean harmful probability: {mean_probability:.6f}")
    print(f"Median harmful probability: {median_probability:.6f}")
    print(f"Minimum harmful probability: {min(probabilities):.6f}")
    print(f"Maximum harmful probability: {max(probabilities):.6f}")
    print(f"BLOCK: {blocked}/{len(category_results)}")
    print(f"ALLOW: {allowed}/{len(category_results)}")
    print(f"BLOCK rate: {blocked / len(category_results):.2%}")
    print(f"ALLOW rate: {allowed / len(category_results):.2%}")


harmful_response_results = categories["harmful_response"]

harmful_response_false_negative_rate = sum(
    1
    for result in harmful_response_results
    if result["decision"] == "ALLOW"
) / len(harmful_response_results)


safe_refusal_results = categories["safe_refusal"]

safe_refusal_false_positive_rate = sum(
    1
    for result in safe_refusal_results
    if result["decision"] == "BLOCK"
) / len(safe_refusal_results)


generic_safe_results = categories["generic_safe_response"]

generic_safe_false_positive_rate = sum(
    1
    for result in generic_safe_results
    if result["decision"] == "BLOCK"
) / len(generic_safe_results)


print("\n" + "=" * 80)
print("KEY SAFETY METRICS")
print("=" * 80)

print(
    f"\nHarmful response false-negative rate: "
    f"{harmful_response_false_negative_rate:.2%}"
)

print(
    f"Safe refusal false-positive rate: "
    f"{safe_refusal_false_positive_rate:.2%}"
)

print(
    f"Generic safe response false-positive rate: "
    f"{generic_safe_false_positive_rate:.2%}"
)


print("\n" + "=" * 80)
print("INDIVIDUAL RESULTS")
print("=" * 80)

for result in results:

    print(
        f"\n[{result['category']}] "
        f"harmful={result['harmful_probability']:.6f} "
        f"decision={result['decision']}"
    )

    print(result["text"])


output = {
    "dataset": {
        "path": dataset_path,
        "total_samples": len(dataset)
    },
    "summary": summary,
    "key_metrics": {
        "harmful_response_false_negative_rate": harmful_response_false_negative_rate,
        "safe_refusal_false_positive_rate": safe_refusal_false_positive_rate,
        "generic_safe_response_false_positive_rate": generic_safe_false_positive_rate
    },
    "results": results
}

os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print("RESULTS SAVED")
print("=" * 80)
print(results_path)