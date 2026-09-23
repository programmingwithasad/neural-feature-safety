import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


TRAIN_DATA_PATH = "data/processed/prompt_activations_10000.pt"
TEST_DATA_PATH = "data/processed/prompt_test_activations.pt"


train_data = torch.load(
    TRAIN_DATA_PATH,
    map_location="cpu"
)

test_data = torch.load(
    TEST_DATA_PATH,
    map_location="cpu"
)

train_features = train_data["features"].numpy()
train_labels = train_data["labels"].numpy()

test_features = test_data["features"].numpy()
test_labels = test_data["labels"].numpy()

print(
    "Training DistilBERT features:",
    train_features.shape
)

print(
    "Testing DistilBERT features:",
    test_features.shape
)

classifier = LogisticRegression(
    max_iter=1000
)

classifier.fit(
    train_features,
    train_labels
)

predictions = classifier.predict(
    test_features
)

accuracy = accuracy_score(
    test_labels,
    predictions
)

print(
    "\nTest Accuracy:",
    accuracy
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        test_labels,
        predictions,
        target_names=[
            "unharmful",
            "harmful"
        ]
    )
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        test_labels,
        predictions
    )
)
