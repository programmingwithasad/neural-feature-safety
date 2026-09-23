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
    "final_threshold_test_results.csv"
)

OLD_THRESHOLD = 0.60
FINAL_THRESHOLD = 0.50


def calculate_metrics(labels, probabilities, threshold):

    predictions = probabilities >= threshold

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
        "threshold": threshold,
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
    print("FINAL THRESHOLD HELD-OUT TEST EVALUATION")
    print("=" * 60)

    data = torch.load(
        FEATURES_PATH,
        map_location="cpu",
        weights_only=False
    )

    features = data["features"].float().numpy()
    labels = data["labels"].numpy()

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    probabilities = classifier.predict_proba(
        features
    )[:, 1]

    print(
        f"Test samples: {len(labels)}"
    )

    print(
        f"Harmful samples: {np.sum(labels == 1)}"
    )

    print(
        f"Benign samples: {np.sum(labels == 0)}"
    )

    old_metrics = calculate_metrics(
        labels,
        probabilities,
        OLD_THRESHOLD
    )

    final_metrics = calculate_metrics(
        labels,
        probabilities,
        FINAL_THRESHOLD
    )

    results = pd.DataFrame(
        [
            old_metrics,
            final_metrics
        ]
    )

    print("\nCurrent threshold = 0.60")

    print(
        f"Accuracy: {old_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {old_metrics['precision']:.4f}"
    )

    print(
        f"Recall: {old_metrics['recall']:.4f}"
    )

    print(
        f"F1: {old_metrics['f1']:.4f}"
    )

    print(
        f"TP: {old_metrics['tp']}"
    )

    print(
        f"TN: {old_metrics['tn']}"
    )

    print(
        f"FP: {old_metrics['fp']}"
    )

    print(
        f"FN: {old_metrics['fn']}"
    )

    print("\nValidation-selected threshold = 0.50")

    print(
        f"Accuracy: {final_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {final_metrics['precision']:.4f}"
    )

    print(
        f"Recall: {final_metrics['recall']:.4f}"
    )

    print(
        f"F1: {final_metrics['f1']:.4f}"
    )

    print(
        f"TP: {final_metrics['tp']}"
    )

    print(
        f"TN: {final_metrics['tn']}"
    )

    print(
        f"FP: {final_metrics['fp']}"
    )

    print(
        f"FN: {final_metrics['fn']}"
    )

    print("\nChanges")

    print(
        f"Recall change: "
        f"{final_metrics['recall'] - old_metrics['recall']:.4f}"
    )

    print(
        f"False positive change: "
        f"{final_metrics['fp'] - old_metrics['fp']}"
    )

    print(
        f"False negative change: "
        f"{final_metrics['fn'] - old_metrics['fn']}"
    )

    print(
        f"F1 change: "
        f"{final_metrics['f1'] - old_metrics['f1']:.4f}"
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