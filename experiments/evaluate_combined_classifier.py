import torch
import pickle
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


FEATURE_PATH = "data/processed/prompt_test_sae_features.pt"
MODEL_PATH = "models/combined_safety_classifier.pkl"


data = torch.load(
    FEATURE_PATH,
    map_location="cpu"
)

X_test = data["features"].numpy()
y_test = data["labels"].numpy()

with open(
    MODEL_PATH,
    "rb"
) as file:

    classifier = pickle.load(
        file
    )

predictions = classifier.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "Test Accuracy:",
    accuracy
)

print(
    "\nClassification Report:"
)

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

print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        predictions
    )
)