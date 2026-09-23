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
    "final_threshold_calibration.csv"
)

THRESHOLDS = np.arange(
    0.40,
    0.701,
    0.01
)


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

    return {
        "threshold": round(
            float(threshold),
            2
        ),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "harmful_block_rate": recall,
        "benign_allow_rate": (
            tn / (tn + fp)
            if tn + fp > 0
            else 0
        )
    }


def main():

    print("=" * 60)
    print("FINAL VALIDATION THRESHOLD CALIBRATION")
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
        f"Validation samples: {len(labels)}"
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

    results_df = pd.DataFrame(
        results
    )

    best_index = results_df[
        "f1"
    ].idxmax()

    best = results_df.loc[
        best_index
    ]

    print("\nThreshold evaluation")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "harmful_block_rate": "{:.4f}".format,
                "benign_allow_rate": "{:.4f}".format
            }
        )
    )

    print("\nSelected threshold")

    print(
        f"Threshold: {best['threshold']:.2f}"
    )

    print(
        f"Accuracy: {best['accuracy']:.4f}"
    )

    print(
        f"Precision: {best['precision']:.4f}"
    )

    print(
        f"Recall: {best['recall']:.4f}"
    )

    print(
        f"F1: {best['f1']:.4f}"
    )

    print(
        f"TP: {int(best['tp'])}"
    )

    print(
        f"TN: {int(best['tn'])}"
    )

    print(
        f"FP: {int(best['fp'])}"
    )

    print(
        f"FN: {int(best['fn'])}"
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