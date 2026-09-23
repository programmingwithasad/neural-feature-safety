import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from autoencoder import SparseAutoencoder


TRAIN_DATA_PATH = "data/processed/prompt_activations_10000.pt"
TEST_DATA_PATH = "data/processed/prompt_test_activations.pt"
SAE_MODEL_PATH = "models/sparse_autoencoder_10000.pt"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_data = torch.load(
    TRAIN_DATA_PATH,
    map_location=device
)

test_data = torch.load(
    TEST_DATA_PATH,
    map_location=device
)

train_features = train_data["features"].float().to(device)
train_labels = train_data["labels"].cpu().numpy()

test_features = test_data["features"].float().to(device)
test_labels = test_data["labels"].cpu().numpy()

sae = SparseAutoencoder(
    input_dim=768,
    hidden_dim=2048
).to(device)

sae.load_state_dict(
    torch.load(
        SAE_MODEL_PATH,
        map_location=device
    )
)

sae.eval()

with torch.no_grad():

    _, train_sparse = sae(
        train_features
    )

    _, test_sparse = sae(
        test_features
    )

train_sparse = train_sparse.cpu().numpy()
test_sparse = test_sparse.cpu().numpy()

print("Training SAE features:", train_sparse.shape)
print("Testing SAE features:", test_sparse.shape)

classifier = LogisticRegression(
    max_iter=1000
)

classifier.fit(
    train_sparse,
    train_labels
)

predictions = classifier.predict(
    test_sparse
)

accuracy = accuracy_score(
    test_labels,
    predictions
)

print("\nTest Accuracy:", accuracy)

print("\nClassification Report:")

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

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        test_labels,
        predictions
    )
)