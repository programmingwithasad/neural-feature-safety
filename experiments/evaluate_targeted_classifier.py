import torch
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


MODEL_PATH = "models/targeted_safety_classifier.pkl"
TEST_PATH = "data/processed/prompt_test_sae_features.pt"


data = torch.load(
    TEST_PATH,
    map_location="cpu"
)

X = data["features"].numpy()

y = data["labels"]

if not torch.is_tensor(y):

    y = torch.tensor(
        [
            1 if str(x).lower() == "harmful" else 0
            for x in y
        ]
    )

y = y.numpy().astype(np.int64)


classifier = joblib.load(
    MODEL_PATH
)

probabilities = classifier.predict_proba(
    X
)[:, 1]

predictions = (
    probabilities >= 0.60
).astype(np.int64)


accuracy = accuracy_score(
    y,
    predictions
)

print(
    "TARGETED CLASSIFIER TEST EVALUATION"
)

print(
    "=" * 70
)

print(
    f"Threshold: 0.60"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print()

print(
    "Classification Report:"
)

print(
    classification_report(
        y,
        predictions,
        target_names=[
            "unharmful",
            "harmful"
        ],
        digits=4
    )
)

print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        y,
        predictions
    )
)

print()

print(
    f"Test samples: {len(y)}"
)

print(
    f"Harmful samples: {(y == 1).sum()}"
)

print(
    f"Unharmful samples: {(y == 0).sum()}"
)
