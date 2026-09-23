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
    "gateway_safety_benchmark.json"
)

results_path = os.path.join(
    project_root,
    "data",
    "processed",
    "gateway_safety_benchmark_results.json"
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
        "topic": sample["topic"],
        "prompt": sample["prompt"],
        "text": sample["text"],
        "expected": sample["expected"],
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


summary = {}

print("=" * 80)
print("GATEWAY SAFETY BENCHMARK EVALUATION")
print("=" * 80)


for category, category_results in categories.items():

    probabilities = [
        result["harmful_probability"]
        for result in category_results
    ]

    correct = sum(
        1
        for result in category_results
        if result["decision"] == result["expected"]
    )

    blocked = sum(
        1
        for result in category_results
        if result["decision"] == "BLOCK"
    )

    allowed = sum(
        1
        for result in category_results
        if result["decision"] == "ALLOW"
    )

    summary[category] = {
        "count": len(category_results),
        "expected": category_results[0]["expected"],
        "correct": correct,
        "accuracy": correct / len(category_results),
        "mean_harmful_probability": statistics.mean(probabilities),
        "median_harmful_probability": statistics.median(probabilities),
        "minimum_harmful_probability": min(probabilities),
        "maximum_harmful_probability": max(probabilities),
        "blocked": blocked,
        "allowed": allowed
    }

    print("\n" + "-" * 80)
    print(category)
    print("-" * 80)

    print(f"Samples: {len(category_results)}")
    print(f"Expected: {category_results[0]['expected']}")
    print(f"Correct: {correct}/{len(category_results)}")
    print(f"Accuracy: {correct / len(category_results):.2%}")
    print(f"Mean harmful probability: {statistics.mean(probabilities):.6f}")
    print(f"Median harmful probability: {statistics.median(probabilities):.6f}")
    print(f"Minimum harmful probability: {min(probabilities):.6f}")
    print(f"Maximum harmful probability: {max(probabilities):.6f}")
    print(f"BLOCK: {blocked}/{len(category_results)}")
    print(f"ALLOW: {allowed}/{len(category_results)}")


harmful_prompt_results = categories["harmful_prompt"]
harmful_response_results = categories["harmful_response"]
safe_refusal_results = categories["safe_refusal"]
generic_safe_results = categories["generic_safe_response"]


harmful_prompt_block_rate = sum(
    result["decision"] == "BLOCK"
    for result in harmful_prompt_results
) / len(harmful_prompt_results)


harmful_response_block_rate = sum(
    result["decision"] == "BLOCK"
    for result in harmful_response_results
) / len(harmful_response_results)


safe_refusal_allow_rate = sum(
    result["decision"] == "ALLOW"
    for result in safe_refusal_results
) / len(safe_refusal_results)


generic_safe_allow_rate = sum(
    result["decision"] == "ALLOW"
    for result in generic_safe_results
) / len(generic_safe_results)


print("\n" + "=" * 80)
print("GATEWAY-RELEVANT METRICS")
print("=" * 80)

print(
    f"\nHarmful prompt BLOCK rate: "
    f"{harmful_prompt_block_rate:.2%}"
)

print(
    f"Harmful response BLOCK rate: "
    f"{harmful_response_block_rate:.2%}"
)

print(
    f"Safe refusal ALLOW rate: "
    f"{safe_refusal_allow_rate:.2%}"
)

print(
    f"Generic safe response ALLOW rate: "
    f"{generic_safe_allow_rate:.2%}"
)


print("\n" + "=" * 80)
print("PAIRED TOPIC COMPARISON")
print("=" * 80)


paired_topics = {}

for result in results:

    if result["category"] not in {
        "harmful_prompt",
        "harmful_response",
        "safe_refusal"
    }:
        continue

    topic = result["topic"]

    if topic not in paired_topics:
        paired_topics[topic] = {}

    paired_topics[topic][result["category"]] = result


for topic, topic_results in paired_topics.items():

    print("\n" + "-" * 80)
    print(topic)
    print("-" * 80)

    prompt_result = topic_results["harmful_prompt"]
    harmful_result = topic_results["harmful_response"]
    refusal_result = topic_results["safe_refusal"]

    print(
        f"Harmful prompt:   "
        f"{prompt_result['harmful_probability']:.6f} "
        f"→ {prompt_result['decision']}"
    )

    print(
        f"Harmful response: "
        f"{harmful_result['harmful_probability']:.6f} "
        f"→ {harmful_result['decision']}"
    )

    print(
        f"Safe refusal:     "
        f"{refusal_result['harmful_probability']:.6f} "
        f"→ {refusal_result['decision']}"
    )


overall_correct = sum(
    result["decision"] == result["expected"]
    for result in results
)

overall_accuracy = overall_correct / len(results)


output = {
    "dataset": {
        "path": dataset_path,
        "total_samples": len(dataset)
    },
    "summary": summary,
    "gateway_metrics": {
        "harmful_prompt_block_rate": harmful_prompt_block_rate,
        "harmful_response_block_rate": harmful_response_block_rate,
        "safe_refusal_allow_rate": safe_refusal_allow_rate,
        "generic_safe_response_allow_rate": generic_safe_allow_rate
    },
    "overall": {
        "correct": overall_correct,
        "total": len(results),
        "accuracy": overall_accuracy
    },
    "results": results
}


os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n" + "=" * 80)
print("OVERALL")
print("=" * 80)

print(
    f"Correct: {overall_correct}/{len(results)}"
)

print(
    f"Accuracy: {overall_accuracy:.2%}"
)

print("\nResults saved to:")
print(results_path)