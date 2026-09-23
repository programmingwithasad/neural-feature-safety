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
    "threshold_options_results.csv"
)

THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


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

    harmful_block_rate = recall

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
        "harmful_block_rate":
            harmful_block_rate,
        "benign_allow_rate":
            benign_allow_rate
    }


def main():

    print("=" * 60)
    print("SAFETY CLASSIFIER THRESHOLD CALIBRATION")
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

    results = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            labels,
            probabilities,
            threshold
        )

        results.append(
            metrics
        )

        print(
            f"\nThreshold={threshold:.2f}"
        )

        print(
            f"Accuracy={metrics['accuracy']:.4f}"
        )

        print(
            f"Precision={metrics['precision']:.4f}"
        )

        print(
            f"Recall={metrics['recall']:.4f}"
        )

        print(
            f"F1={metrics['f1']:.4f}"
        )

        print(
            f"TP={metrics['tp']} "
            f"TN={metrics['tn']} "
            f"FP={metrics['fp']} "
            f"FN={metrics['fn']}"
        )

        print(
            f"Harmful block rate="
            f"{metrics['harmful_block_rate']:.4f}"
        )

        print(
            f"Benign allow rate="
            f"{metrics['benign_allow_rate']:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nResults:")
    print(
        results_df.to_string(
            index=False
        )
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