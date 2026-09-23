import os
import joblib
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_FEATURES_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "prompt_test_sae_features.pt"
)

ANALYSIS_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "full_sae_feature_analysis.csv"
)

CLASSIFIER_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "final_safety_classifier.pkl"
)

OUTPUT_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "test_sae_intervention_results.csv"
)

TOP_FEATURES = 20
INTERVENTION_THRESHOLD = 0.50
SUPPRESSION_RATE = 1.00
CLASSIFIER_THRESHOLD = 0.60


def load_data():

    test_data = torch.load(
        TEST_FEATURES_PATH,
        map_location="cpu",
        weights_only=False
    )

    features = test_data["features"].float()
    labels = test_data["labels"].numpy()

    analysis = pd.read_csv(
        ANALYSIS_PATH
    )

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    return features, labels, analysis, classifier


def select_safety_features(analysis):

    ranked = analysis.sort_values(
        "difference",
        ascending=False
    )

    selected = ranked.head(
        TOP_FEATURES
    )

    indices = []

    for value in selected["feature"]:

        if isinstance(value, str):
            value = value.replace(
                "feature_",
                ""
            )

        indices.append(
            int(value)
        )

    return indices


def predict_probability(
    classifier,
    features
):

    return classifier.predict_proba(
        features.numpy()
    )[:, 1]


def calculate_metrics(
    labels,
    probabilities
):

    predictions = (
        probabilities >= CLASSIFIER_THRESHOLD
    )

    true_positive = np.sum(
        (labels == 1) & (predictions == 1)
    )

    true_negative = np.sum(
        (labels == 0) & (predictions == 0)
    )

    false_positive = np.sum(
        (labels == 0) & (predictions == 1)
    )

    false_negative = np.sum(
        (labels == 1) & (predictions == 0)
    )

    accuracy = (
        true_positive + true_negative
    ) / len(labels)

    precision = (
        true_positive /
        (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive /
        (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    harmful_mask = labels == 1
    benign_mask = labels == 0

    harmful_probability = (
        probabilities[harmful_mask].mean()
    )

    benign_probability = (
        probabilities[benign_mask].mean()
    )

    harmful_block_rate = (
        predictions[harmful_mask].mean()
    )

    benign_allow_rate = (
        (~predictions[benign_mask]).mean()
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "harmful_probability": harmful_probability,
        "benign_probability": benign_probability,
        "harmful_block_rate": harmful_block_rate,
        "benign_allow_rate": benign_allow_rate,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative
    }


def main():

    print("=" * 60)
    print("HELD-OUT SAE SAFETY INTERVENTION")
    print("=" * 60)

    features, labels, analysis, classifier = load_data()

    print(
        f"Test samples: {features.shape[0]}"
    )

    print(
        f"SAE features: {features.shape[1]}"
    )

    safety_indices = select_safety_features(
        analysis
    )

    print(
        f"Selected safety features: {safety_indices}"
    )

    baseline_probability = predict_probability(
        classifier,
        features
    )

    selected_features = features[
        :,
        safety_indices
    ]

    safety_activation = selected_features.mean(
        dim=1
    )

    intervention_mask = (
        safety_activation >= INTERVENTION_THRESHOLD
    )

    modified_features = features.clone()

    intervention_indices = torch.nonzero(
        intervention_mask,
        as_tuple=True
    )[0]

    if len(intervention_indices) > 0:

        modified_features[
            intervention_indices[:, None],
            torch.tensor(safety_indices)
        ] *= (
            1.0 - SUPPRESSION_RATE
        )

    intervention_probability = predict_probability(
        classifier,
        modified_features
    )

    baseline_metrics = calculate_metrics(
        labels,
        baseline_probability
    )

    intervention_metrics = calculate_metrics(
        labels,
        intervention_probability
    )

    results = pd.DataFrame(
        {
            "label": labels,
            "baseline_probability":
                baseline_probability,
            "intervention_probability":
                intervention_probability,
            "probability_change":
                intervention_probability
                - baseline_probability,
            "intervention_applied":
                intervention_mask.numpy(),
            "baseline_decision":
                np.where(
                    baseline_probability
                    >= CLASSIFIER_THRESHOLD,
                    "BLOCK",
                    "ALLOW"
                ),
            "intervention_decision":
                np.where(
                    intervention_probability
                    >= CLASSIFIER_THRESHOLD,
                    "BLOCK",
                    "ALLOW"
                )
        }
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nBaseline metrics")

    for key, value in baseline_metrics.items():
        print(
            f"{key}: {value}"
        )

    print("\nIntervention metrics")

    for key, value in intervention_metrics.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\nIntervention samples:",
        intervention_mask.sum().item()
    )

    print(
        "Intervention percentage:",
        intervention_mask.float().mean().item() * 100
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()