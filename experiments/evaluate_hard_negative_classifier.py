import torch
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


MODEL_PATH = "models/hard_negative_safety_classifier.pkl"
TEST_PATH = "data/processed/prompt_test_sae_features.pt"


data = torch.load(
    TEST_PATH,
    map_location="cpu"
)

X_test = data["features"].numpy()
y_test = data["labels"].numpy().astype(np.int64)


classifier = joblib.load(
    MODEL_PATH
)

probabilities = classifier.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(np.int64)


accuracy = accuracy_score(
    y_test,
    predictions
)

print("HARD-NEGATIVE CLASSIFIER TEST EVALUATION")
print("=" * 60)
print(f"Test Accuracy: {accuracy:.4f}")
print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "unharmful",
            "harmful"
        ]
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)