import json
import os

import joblib
import numpy as np
import torch


project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

production_path = os.path.join(
    project_root,
    "models",
    "final_safety_classifier.pkl"
)

candidate_path = os.path.join(
    project_root,
    "models",
    "candidate_safety_classifier_refusal_v1.pkl"
)

features_path = os.path.join(
    project_root,
    "data",
    "processed",
    "independent_gateway_benchmark_features.pt"
)

benchmark_path = os.path.join(
    project_root,
    "data",
    "processed",
    "independent_gateway_benchmark.json"
)

output_path = os.path.join(
    project_root,
    "data",
    "processed",
    "independent_gateway_benchmark_results.json"
)


production = joblib.load(
    production_path
)

candidate = joblib.load(
    candidate_path
)


data = torch.load(
    features_path,
    map_location="cpu"
)

features = data["features"]

if torch.is_tensor(features):
    features = features.numpy()

labels = data["labels"]

if torch.is_tensor(labels):
    labels = labels.numpy()

with open(
    benchmark_path,
    "r",
    encoding="utf-8"
) as f:
    benchmark = json.load(f)

examples = benchmark["examples"]


def evaluate_model(model):

    probabilities = model.predict_proba(
        features
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(np.int64)

    results = []

    for index, example in enumerate(examples):

        expected = example["expected"]

        actual = (
            "BLOCK"
            if predictions[index] == 1
            else "ALLOW"
        )

        results.append({
            "id": example["id"],
            "category": example["category"],
            "topic": example["topic"],
            "text": example["text"],
            "expected": expected,
            "decision": actual,
            "harmful_probability": float(
                probabilities[index]
            ),
            "correct": actual == expected
        })

    return results


def summarize(results):

    total = len(results)

    correct = sum(
        result["correct"]
        for result in results
    )

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "categories": {}
    }

    categories = sorted(
        set(
            result["category"]
            for result in results
        )
    )

    for category in categories:

        category_results = [
            result
            for result in results
            if result["category"] == category
        ]

        category_total = len(
            category_results
        )

        category_correct = sum(
            result["correct"]
            for result in category_results
        )

        probabilities = [
            result["harmful_probability"]
            for result in category_results
        ]

        blocked = sum(
            result["decision"] == "BLOCK"
            for result in category_results
        )

        allowed = sum(
            result["decision"] == "ALLOW"
            for result in category_results
        )

        summary["categories"][category] = {
            "total": category_total,
            "correct": category_correct,
            "accuracy": (
                category_correct
                / category_total
            ),
            "blocked": blocked,
            "allowed": allowed,
            "mean_harmful_probability": float(
                np.mean(probabilities)
            ),
            "median_harmful_probability": float(
                np.median(probabilities)
            )
        }

    return summary


print("=" * 80)
print("INDEPENDENT GATEWAY BENCHMARK")
print("PRODUCTION VS CANDIDATE V1")
print("=" * 80)


production_results = evaluate_model(
    production
)

candidate_results = evaluate_model(
    candidate
)


production_summary = summarize(
    production_results
)

candidate_summary = summarize(
    candidate_results
)


print("\n" + "-" * 80)
print("OVERALL")
print("-" * 80)

print(
    "\nProduction:"
)

print(
    f"Accuracy: "
    f"{production_summary['accuracy']:.2%}"
)

print(
    "\nCandidate v1:"
)

print(
    f"Accuracy: "
    f"{candidate_summary['accuracy']:.2%}"
)


print("\n" + "-" * 80)
print("CATEGORY RESULTS")
print("-" * 80)


for category in sorted(
    production_summary["categories"]
):

    production_category = (
        production_summary["categories"][category]
    )

    candidate_category = (
        candidate_summary["categories"][category]
    )

    print(
        f"\n{category}"
    )

    print(
        "Production: "
        f"accuracy="
        f"{production_category['accuracy']:.2%}, "
        f"BLOCK="
        f"{production_category['blocked']}/"
        f"{production_category['total']}, "
        f"mean="
        f"{production_category['mean_harmful_probability']:.6f}"
    )

    print(
        "Candidate:  "
        f"accuracy="
        f"{candidate_category['accuracy']:.2%}, "
        f"BLOCK="
        f"{candidate_category['blocked']}/"
        f"{candidate_category['total']}, "
        f"mean="
        f"{candidate_category['mean_harmful_probability']:.6f}"
    )


production_by_id = {
    result["id"]: result
    for result in production_results
}

candidate_by_id = {
    result["id"]: result
    for result in candidate_results
}


disagreements = []

for example_id in production_by_id:

    production_result = (
        production_by_id[example_id]
    )

    candidate_result = (
        candidate_by_id[example_id]
    )

    if (
        production_result["decision"]
        != candidate_result["decision"]
    ):

        disagreements.append({
            "id": example_id,
            "category": production_result["category"],
            "topic": production_result["topic"],
            "text": production_result["text"],
            "expected": production_result["expected"],
            "production_decision": production_result["decision"],
            "candidate_decision": candidate_result["decision"],
            "production_probability": production_result[
                "harmful_probability"
            ],
            "candidate_probability": candidate_result[
                "harmful_probability"
            ],
            "production_correct": production_result[
                "correct"
            ],
            "candidate_correct": candidate_result[
                "correct"
            ]
        })


print("\n" + "-" * 80)
print("DECISION DISAGREEMENTS")
print("-" * 80)

print(
    f"\nTotal disagreements: "
    f"{len(disagreements)}"
)


for disagreement in disagreements:

    print(
        "\n"
        + "-" * 60
    )

    print(
        f"ID: "
        f"{disagreement['id']}"
    )

    print(
        f"Category: "
        f"{disagreement['category']}"
    )

    print(
        f"Topic: "
        f"{disagreement['topic']}"
    )

    print(
        f"Expected: "
        f"{disagreement['expected']}"
    )

    print(
        f"Production: "
        f"{disagreement['production_decision']} "
        f"({disagreement['production_probability']:.6f})"
    )

    print(
        f"Candidate: "
        f"{disagreement['candidate_decision']} "
        f"({disagreement['candidate_probability']:.6f})"
    )

    print(
        f"Production correct: "
        f"{disagreement['production_correct']}"
    )

    print(
        f"Candidate correct: "
        f"{disagreement['candidate_correct']}"
    )

    print(
        f"Text: "
        f"{disagreement['text']}"
    )


output = {
    "benchmark": {
        "name": benchmark[
            "benchmark_name"
        ],
        "total_examples": len(examples),
        "category_counts": benchmark[
            "category_counts"
        ]
    },
    "production": production_summary,
    "candidate_v1": candidate_summary,
    "production_results": production_results,
    "candidate_results": candidate_results,
    "decision_disagreements": disagreements
}


with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n" + "=" * 80)
print("EVALUATION SAVED")
print("=" * 80)

print(output_path)