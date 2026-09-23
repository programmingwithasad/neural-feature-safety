import torch
import pickle
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


FEATURE_PATH = "data/processed/prompt_validation_sae_features.pt"
MODEL_PATH = "models/combined_safety_classifier.pkl"


data = torch.load(
    FEATURE_PATH,
    map_location="cpu"
)

X_validation = data["features"].numpy()
y_validation = data["labels"].numpy()

with open(
    MODEL_PATH,
    "rb"
) as file:

    classifier = pickle.load(
        file
    )

probabilities = classifier.predict_proba(
    X_validation
)[:, 1]

print(
    "Combined classifier threshold calibration"
)

print(
    "Validation samples:",
    len(y_validation)
)

print(
    "Harmful samples:",
    np.sum(y_validation == 1)
)

print(
    "Unharmful samples:",
    np.sum(y_validation == 0)
)

print(
    "\nThreshold Analysis"
)

print(
    "=" * 60
)

for threshold in np.arange(
    0.05,
    1.00,
    0.05
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_validation,
        predictions
    )

    tn, fp, fn, tp = matrix.ravel()

    false_positive_rate = (
        fp / (fp + tn)
    )

    false_negative_rate = (
        fn / (fn + tp)
    )

    print(
        f"Threshold: {threshold:.2f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{false_positive_rate:.4f}"
    )

    print(
        f"False Negative Rate: "
        f"{false_negative_rate:.4f}"
    )

    print()