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
    "prompt_validation_sae_features.pt"
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
    "conditional_feature_850_validation.csv"
)

FINAL_THRESHOLD = 0.50
FEATURE_INDEX = 850

CONDITION_THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

SUPPRESSION_RATES = [
    0.05,
    0.10,
    0.15,
    0.20
]


def calculate_metrics(labels, probabilities):

    predictions = probabilities >= FINAL_THRESHOLD

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

    benign_allow_rate = (
        tn / (tn + fp)
        if tn + fp > 0
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
        "fn": fn,
        "harmful_block_rate": recall,
        "benign_allow_rate": benign_allow_rate
    }


def main():

    print("=" * 60)
    print("CONDITIONAL FEATURE 850 VALIDATION")
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

    original_features = features.numpy()

    baseline_probabilities = classifier.predict_proba(
        original_features
    )[:, 1]

    baseline_metrics = calculate_metrics(
        labels,
        baseline_probabilities
    )

    print(
        f"Validation samples: {len(labels)}"
    )

    print(
        f"Harmful samples: {np.sum(labels == 1)}"
    )

    print(
        f"Benign samples: {np.sum(labels == 0)}"
    )

    print("\nBaseline threshold = 0.50")

    print(
        f"Accuracy: {baseline_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {baseline_metrics['precision']:.4f}"
    )

    print(
        f"Recall: {baseline_metrics['recall']:.4f}"
    )

    print(
        f"F1: {baseline_metrics['f1']:.4f}"
    )

    print(
        f"FP: {baseline_metrics['fp']}"
    )

    print(
        f"FN: {baseline_metrics['fn']}"
    )

    results = []

    for condition_threshold in CONDITION_THRESHOLDS:

        for suppression_rate in SUPPRESSION_RATES:

            intervention_features = features.clone()

            intervention_mask = (
                baseline_probabilities <
                condition_threshold
            )

            intervention_features[
                intervention_mask,
                FEATURE_INDEX
            ] *= (
                1.0 - suppression_rate
            )

            intervention_probabilities = classifier.predict_proba(
                intervention_features.numpy()
            )[:, 1]

            metrics = calculate_metrics(
                labels,
                intervention_probabilities
            )

            baseline_predictions = (
                baseline_probabilities >= FINAL_THRESHOLD
            )

            intervention_predictions = (
                intervention_probabilities >= FINAL_THRESHOLD
            )

            harmful_mask = labels == 1
            benign_mask = labels == 0

            harmful_blocked = np.sum(
                intervention_predictions &
                harmful_mask
            )

            harmful_total = np.sum(
                harmful_mask
            )

            harmful_allow_count = (
                harmful_total -
                harmful_blocked
            )

            benign_allowed = np.sum(
                (~intervention_predictions) &
                benign_mask
            )

            benign_total = np.sum(
                benign_mask
            )

            intervention_count = np.sum(
                intervention_mask
            )

            harmful_intervened = np.sum(
                intervention_mask &
                harmful_mask
            )

            benign_intervened = np.sum(
                intervention_mask &
                benign_mask
            )

            harmful_became_allowed = np.sum(
                baseline_predictions &
                harmful_mask &
                ~intervention_predictions
            )

            benign_became_blocked = np.sum(
                ~baseline_predictions &
                benign_mask &
                intervention_predictions
            )

            decision_changes = np.sum(
                baseline_predictions !=
                intervention_predictions
            )

            results.append(
                {
                    "condition_threshold":
                        condition_threshold,
                    "suppression_rate":
                        suppression_rate,
                    "intervention_samples":
                        intervention_count,
                    "intervention_percentage":
                        intervention_count /
                        len(labels) *
                        100,
                    "harmful_intervened":
                        harmful_intervened,
                    "benign_intervened":
                        benign_intervened,
                    "harmful_became_allowed":
                        harmful_became_allowed,
                    "benign_became_blocked":
                        benign_became_blocked,
                    "decision_changes":
                        decision_changes,
                    **metrics
                }
            )

    results_df = pd.DataFrame(
        results
    )

    results_df["fp_change"] = (
        results_df["fp"] -
        baseline_metrics["fp"]
    )

    results_df["fn_change"] = (
        results_df["fn"] -
        baseline_metrics["fn"]
    )

    results_df["f1_change"] = (
        results_df["f1"] -
        baseline_metrics["f1"]
    )

    safe_results = results_df[
        results_df["harmful_became_allowed"] == 0
    ].copy()

    safe_results = safe_results.sort_values(
        [
            "f1",
            "precision",
            "accuracy"
        ],
        ascending=False
    )

    print("\nBest configurations with zero harmful BLOCK → ALLOW changes")

    if len(safe_results) > 0:

        print(
            safe_results[
                [
                    "condition_threshold",
                    "suppression_rate",
                    "intervention_samples",
                    "intervention_percentage",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "fp",
                    "fn",
                    "fp_change",
                    "fn_change",
                    "harmful_became_allowed",
                    "benign_became_blocked",
                    "decision_changes"
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

    else:

        print(
            "No configuration satisfied the zero harmful BLOCK → ALLOW constraint."
        )

    results_df.to_csv(
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