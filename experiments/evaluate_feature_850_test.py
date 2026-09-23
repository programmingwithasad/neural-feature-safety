import os
import joblib
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "prompt_test_sae_features.pt"
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
    "feature_850_test_intervention.csv"
)

FEATURE_INDEX = 850
SUPPRESSION_RATE = 0.20
CLASSIFIER_THRESHOLD = 0.60


def load_data():

    data = torch.load(
        FEATURES_PATH,
        map_location="cpu",
        weights_only=False
    )

    features = data["features"].float()
    labels = data["labels"].numpy()

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    return features, labels, classifier


def predict(
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

    tp = np.sum(
        (labels == 1) & predictions
    )

    tn = np.sum(
        (labels == 0) & ~predictions
    )

    fp = np.sum(
        (labels == 0) & predictions
    )

    fn = np.sum(
        (labels == 1) & ~predictions
    )

    accuracy = (
        tp + tn
    ) / len(labels)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def main():

    print("=" * 60)
    print("FEATURE 850 HELD-OUT TEST INTERVENTION")
    print("=" * 60)

    features, labels, classifier = load_data()

    print(
        f"Test samples: {len(labels)}"
    )

    baseline_probability = predict(
        classifier,
        features
    )

    modified_features = features.clone()

    modified_features[:, FEATURE_INDEX] *= (
        1.0 - SUPPRESSION_RATE
    )

    intervention_probability = predict(
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

    print("\nBaseline")

    for key, value in baseline_metrics.items():

        print(
            f"{key}: {value}"
        )

    print("\nFeature 850 intervention")

    for key, value in intervention_metrics.items():

        print(
            f"{key}: {value}"
        )

    print("\nChanges")

    print(
        "False positive change:",
        intervention_metrics["fp"]
        - baseline_metrics["fp"]
    )

    print(
        "False negative change:",
        intervention_metrics["fn"]
        - baseline_metrics["fn"]
    )

    print(
        "True positive change:",
        intervention_metrics["tp"]
        - baseline_metrics["tp"]
    )

    print(
        "True negative change:",
        intervention_metrics["tn"]
        - baseline_metrics["tn"]
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
                - baseline_probability
        }
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()