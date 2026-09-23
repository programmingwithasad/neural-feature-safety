import json
import os

import joblib
import numpy as np
import torch

import sys


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

benchmark_path = os.path.join(
    project_root,
    "data",
    "processed",
    "gateway_safety_benchmark_features.pt"
)

test_path = os.path.join(
    project_root,
    "data",
    "processed",
    "prompt_test_sae_features.pt"
)

validation_path = os.path.join(
    project_root,
    "data",
    "processed",
    "prompt_validation_sae_features.pt"
)

diagnostic_path = os.path.join(
    project_root,
    "data",
    "processed",
    "refusal_diagnostic_results.json"
)

output_path = os.path.join(
    project_root,
    "data",
    "processed",
    "production_vs_candidate_v1.json"
)


production = joblib.load(
    production_path
)

candidate = joblib.load(
    candidate_path
)


def load_features(path):

    data = torch.load(
        path,
        map_location="cpu"
    )

    features = data["features"]

    if torch.is_tensor(features):
        features = features.numpy()
    else:
        features = np.asarray(features)

    labels = data["labels"]

    if torch.is_tensor(labels):
        labels = labels.numpy()
    else:
        labels = np.asarray(labels)

        if labels.dtype.kind not in "iu":
            labels = np.array(
                [
                    1 if str(x).lower() == "harmful" else 0
                    for x in labels
                ],
                dtype=np.int64
            )

    return features, labels.astype(np.int64)


def evaluate_model(model, features, labels):

    probabilities = model.predict_proba(features)[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(np.int64)

    accuracy = (
        predictions == labels
    ).mean()

    harmful_mask = labels == 1
    safe_mask = labels == 0

    harmful_count = int(
        harmful_mask.sum()
    )

    safe_count = int(
        safe_mask.sum()
    )

    harmful_recall = (
        (
            predictions[harmful_mask] == 1
        ).mean()
        if harmful_count > 0
        else 0.0
    )

    safe_recall = (
        (
            predictions[safe_mask] == 0
        ).mean()
        if safe_count > 0
        else 0.0
    )

    false_positive_rate = 1.0 - safe_recall

    false_negative_rate = 1.0 - harmful_recall

    return {
        "accuracy": float(accuracy),
        "harmful_recall": float(harmful_recall),
        "safe_recall": float(safe_recall),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "mean_harmful_probability": float(
            probabilities.mean()
        ),
        "median_harmful_probability": float(
            np.median(probabilities)
        )
    }


def evaluate_benchmark(model, features, metadata):

    probabilities = model.predict_proba(
        features
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    )

    results = []

    for index, probability in enumerate(
        probabilities
    ):

        expected = metadata[index]["expected"]

        decision = (
            "BLOCK"
            if predictions[index]
            else "ALLOW"
        )

        results.append({
            "category": metadata[index]["category"],
            "topic": metadata[index]["topic"],
            "expected": expected,
            "decision": decision,
            "harmful_probability": float(
                probability
            ),
            "correct": decision == expected
        })

    return results


def summarize_benchmark(results):

    summary = {}

    for result in results:

        category = result["category"]

        if category not in summary:
            summary[category] = {
                "total": 0,
                "correct": 0,
                "blocked": 0,
                "allowed": 0,
                "probabilities": []
            }

        entry = summary[category]

        entry["total"] += 1

        if result["correct"]:
            entry["correct"] += 1

        if result["decision"] == "BLOCK":
            entry["blocked"] += 1
        else:
            entry["allowed"] += 1

        entry["probabilities"].append(
            result["harmful_probability"]
        )

    for category, entry in summary.items():

        probabilities = entry.pop(
            "probabilities"
        )

        entry["accuracy"] = (
            entry["correct"]
            / entry["total"]
        )

        entry["mean_harmful_probability"] = float(
            np.mean(probabilities)
        )

        entry["median_harmful_probability"] = float(
            np.median(probabilities)
        )

    return summary


print("=" * 80)
print("PRODUCTION VS CANDIDATE SAFETY CLASSIFIER")
print("=" * 80)


test_features, test_labels = load_features(
    test_path
)

validation_features, validation_labels = load_features(
    validation_path
)

benchmark_data = torch.load(
    benchmark_path,
    map_location="cpu"
)

benchmark_features = benchmark_data[
    "features"
].numpy()

with open(
    os.path.join(
        project_root,
        "data",
        "processed",
        "gateway_safety_benchmark.json"
    ),
    "r",
    encoding="utf-8"
) as f:
    benchmark_metadata = json.load(f)


print("\n" + "-" * 80)
print("EXISTING TEST SET")
print("-" * 80)

production_test = evaluate_model(
    production,
    test_features,
    test_labels
)

candidate_test = evaluate_model(
    candidate,
    test_features,
    test_labels
)

print("\nProduction:")

for key, value in production_test.items():
    print(f"{key}: {value:.6f}")

print("\nCandidate v1:")

for key, value in candidate_test.items():
    print(f"{key}: {value:.6f}")


print("\n" + "-" * 80)
print("EXISTING VALIDATION SET")
print("-" * 80)

production_validation = evaluate_model(
    production,
    validation_features,
    validation_labels
)

candidate_validation = evaluate_model(
    candidate,
    validation_features,
    validation_labels
)

print("\nProduction:")

for key, value in production_validation.items():
    print(f"{key}: {value:.6f}")

print("\nCandidate v1:")

for key, value in candidate_validation.items():
    print(f"{key}: {value:.6f}")


print("\n" + "-" * 80)
print("GATEWAY BENCHMARK")
print("-" * 80)

production_benchmark = evaluate_benchmark(
    production,
    benchmark_features,
    benchmark_metadata
)

candidate_benchmark = evaluate_benchmark(
    candidate,
    benchmark_features,
    benchmark_metadata
)


production_benchmark_summary = summarize_benchmark(
    production_benchmark
)

candidate_benchmark_summary = summarize_benchmark(
    candidate_benchmark
)


for category in production_benchmark_summary:

    print(
        f"\n{category}"
    )

    production_result = (
        production_benchmark_summary[category]
    )

    candidate_result = (
        candidate_benchmark_summary[category]
    )

    print(
        "Production: "
        f"accuracy={production_result['accuracy']:.2%}, "
        f"mean_probability="
        f"{production_result['mean_harmful_probability']:.6f}, "
        f"BLOCK="
        f"{production_result['blocked']}/"
        f"{production_result['total']}"
    )

    print(
        "Candidate:  "
        f"accuracy={candidate_result['accuracy']:.2%}, "
        f"mean_probability="
        f"{candidate_result['mean_harmful_probability']:.6f}, "
        f"BLOCK="
        f"{candidate_result['blocked']}/"
        f"{candidate_result['total']}"
    )


print("\n" + "-" * 80)
print("SAFE-REFUSAL DIAGNOSTIC")
print("-" * 80)


with open(
    diagnostic_path,
    "r",
    encoding="utf-8"
) as f:
    diagnostic = json.load(f)


diagnostic_texts = [
    result["text"]
    for result in diagnostic["results"]
]

diagnostic_categories = [
    result["category"]
    for result in diagnostic["results"]
]


if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import AutoTokenizer, AutoModel
from src.autoencoder import SparseAutoencoder


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

encoder = AutoModel.from_pretrained(
    "distilbert-base-uncased"
)

encoder = encoder.to(device)
encoder.eval()

sae = SparseAutoencoder(
    input_dim=768,
    hidden_dim=2048
)

sae_path = os.path.join(
    project_root,
    "models",
    "sparse_autoencoder_full.pt"
)

checkpoint = torch.load(
    sae_path,
    map_location=device
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    sae.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    sae.load_state_dict(checkpoint)

sae = sae.to(device)
sae.eval()


diagnostic_features = []

with torch.no_grad():

    for start in range(
        0,
        len(diagnostic_texts),
        16
    ):

        batch_texts = diagnostic_texts[
            start:start + 16
        ]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        outputs = encoder(
            **inputs
        )

        hidden = outputs.last_hidden_state

        mask = inputs[
            "attention_mask"
        ].unsqueeze(-1)

        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        _, features = sae(
            pooled
        )

        diagnostic_features.append(
            features.cpu()
        )


diagnostic_features = torch.cat(
    diagnostic_features,
    dim=0
).numpy()


production_probabilities = production.predict_proba(
    diagnostic_features
)[:, 1]

candidate_probabilities = candidate.predict_proba(
    diagnostic_features
)[:, 1]


diagnostic_summary = {}

for category in sorted(
    set(diagnostic_categories)
):

    indices = [
        index
        for index, value in enumerate(
            diagnostic_categories
        )
        if value == category
    ]

    production_values = (
        production_probabilities[indices]
    )

    candidate_values = (
        candidate_probabilities[indices]
    )

    diagnostic_summary[category] = {
        "count": len(indices),
        "production_mean": float(
            np.mean(production_values)
        ),
        "candidate_mean": float(
            np.mean(candidate_values)
        ),
        "production_block_rate": float(
            np.mean(
                production_values >= 0.50
            )
        ),
        "candidate_block_rate": float(
            np.mean(
                candidate_values >= 0.50
            )
        )
    }

    print(
        f"\n{category}"
    )

    print(
        "Production: "
        f"mean={np.mean(production_values):.6f}, "
        f"BLOCK="
        f"{np.mean(production_values >= 0.50):.2%}"
    )

    print(
        "Candidate:  "
        f"mean={np.mean(candidate_values):.6f}, "
        f"BLOCK="
        f"{np.mean(candidate_values >= 0.50):.2%}"
    )


output = {
    "test": {
        "production": production_test,
        "candidate": candidate_test
    },
    "validation": {
        "production": production_validation,
        "candidate": candidate_validation
    },
    "gateway_benchmark": {
        "production": production_benchmark_summary,
        "candidate": candidate_benchmark_summary,
        "production_results": production_benchmark,
        "candidate_results": candidate_benchmark
    },
    "refusal_diagnostic": diagnostic_summary
}


with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        indent=2
    )


print("\n" + "=" * 80)
print("COMPARISON SAVED")
print("=" * 80)

print(output_path)