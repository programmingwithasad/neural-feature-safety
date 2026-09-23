import os
import joblib
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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
    "final_pipeline_test_results.csv"
)

THRESHOLD = 0.50
FEATURE_INDEX = 850
SUPPRESSION_RATE = 0.20


def calculate_metrics(labels, probabilities):

    predictions = probabilities >= THRESHOLD

    tp = int(
        np.sum(
            (labels == 1) & predictions
        )
    )

    tn = int(
        np.sum(
            (labels == 0) & ~predictions
        )
    )

    fp = int(
        np.sum(
            (labels == 0) & predictions
        )
    )

    fn = int(
        np.sum(
            (labels == 1) & ~predictions
        )
    )

    accuracy = (
        (tp + tn) / len(labels)
    )

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
    print("FINAL PIPELINE HELD-OUT TEST EVALUATION")
    print("=" * 60)

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

    baseline_features = features.numpy()

    baseline_probabilities = classifier.predict_proba(
        baseline_features
    )[:, 1]

    baseline_metrics = calculate_metrics(
        labels,
        baseline_probabilities
    )

    intervention_features = features.clone()

    activation = intervention_features[
        :,
        FEATURE_INDEX
    ]

    intervention_features[
        :,
        FEATURE_INDEX
    ] = activation * (
        1 - SUPPRESSION_RATE
    )

    intervention_probabilities = classifier.predict_proba(
        intervention_features.numpy()
    )[:, 1]

    intervention_metrics = calculate_metrics(
        labels,
        intervention_probabilities
    )

    baseline_predictions = (
        baseline_probabilities >= THRESHOLD
    )

    intervention_predictions = (
        intervention_probabilities >= THRESHOLD
    )

    intervention_samples = (
        np.abs(
            intervention_probabilities -
            baseline_probabilities
        ) > 1e-8
    )

    decision_changes = np.sum(
        baseline_predictions !=
        intervention_predictions
    )

    print(
        f"Test samples: {len(labels)}"
    )

    print(
        f"Harmful samples: {np.sum(labels == 1)}"
    )

    print(
        f"Benign samples: {np.sum(labels == 0)}"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    print(
        f"Intervention feature: {FEATURE_INDEX}"
    )

    print(
        f"Suppression rate: {SUPPRESSION_RATE}"
    )

    print("\nBaseline")

    for key, value in baseline_metrics.items():

        if isinstance(value, float):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print("\nFeature 850 + Threshold 0.50")

    for key, value in intervention_metrics.items():

        if isinstance(value, float):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print("\nChanges")

    print(
        f"Accuracy change: "
        f"{intervention_metrics['accuracy'] - baseline_metrics['accuracy']:.6f}"
    )

    print(
        f"Precision change: "
        f"{intervention_metrics['precision'] - baseline_metrics['precision']:.6f}"
    )

    print(
        f"Recall change: "
        f"{intervention_metrics['recall'] - baseline_metrics['recall']:.6f}"
    )

    print(
        f"F1 change: "
        f"{intervention_metrics['f1'] - baseline_metrics['f1']:.6f}"
    )

    print(
        f"False positive change: "
        f"{intervention_metrics['fp'] - baseline_metrics['fp']}"
    )

    print(
        f"False negative change: "
        f"{intervention_metrics['fn'] - baseline_metrics['fn']}"
    )

    print(
        f"True positive change: "
        f"{intervention_metrics['tp'] - baseline_metrics['tp']}"
    )

    print(
        f"True negative change: "
        f"{intervention_metrics['tn'] - baseline_metrics['tn']}"
    )

    print(
        f"Intervention samples: "
        f"{np.sum(intervention_samples)}"
    )

    print(
        f"Decision changes: "
        f"{decision_changes}"
    )

    results = pd.DataFrame(
        [
            {
                "configuration": "threshold_0.50",
                "threshold": THRESHOLD,
                "feature": None,
                "suppression_rate": 0.0,
                **baseline_metrics
            },
            {
                "configuration": "feature_850_threshold_0.50",
                "threshold": THRESHOLD,
                "feature": FEATURE_INDEX,
                "suppression_rate": SUPPRESSION_RATE,
                **intervention_metrics
            }
        ]
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print("=" * 60)


if __name__ == "__main__":
    main()