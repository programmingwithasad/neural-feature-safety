import torch
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


FEATURE_PATH = "data/processed/prompt_test_sae_features.pt"
MODEL_PATH = "models/final_safety_classifier.pkl"

THRESHOLD = 0.60


data = torch.load(
    FEATURE_PATH,
    map_location="cpu"
)

X_test = data["features"].numpy()
y_test = data["labels"].numpy()

classifier = joblib.load(
    MODEL_PATH
)

probabilities = classifier.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
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
    "FINAL TEST EVALUATION"
)

print(
    "=" * 60
)

print(
    f"Threshold: {THRESHOLD:.2f}"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Harmful Precision: {precision:.4f}"
)

print(
    f"Harmful Recall: {recall:.4f}"
)

print(
    f"Harmful F1: {f1:.4f}"
)

print(
    f"False Positive Rate: {false_positive_rate:.4f}"
)

print(
    f"False Negative Rate: {false_negative_rate:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(
    matrix
)

print(
    "\nTest samples:",
    len(y_test)
)

print(
    "Harmful samples:",
    np.sum(y_test == 1)
)

print(
    "Unharmful samples:",
    np.sum(y_test == 0)
)