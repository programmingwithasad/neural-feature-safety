import torch
import joblib

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from autoencoder import SparseAutoencoder


VALIDATION_PATH = "data/processed/prompt_validation_activations.pt"
SAE_PATH = "models/sparse_autoencoder_full.pt"
CLASSIFIER_PATH = "models/safety_classifier.pkl"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = torch.load(
    VALIDATION_PATH,
    map_location="cpu"
)

features = data["features"].float().to(device)
labels = data["labels"].numpy()

model = SparseAutoencoder(
    input_dim=768,
    hidden_dim=2048
).to(device)

model.load_state_dict(
    torch.load(
        SAE_PATH,
        map_location=device
    )
)

model.eval()

classifier = joblib.load(
    CLASSIFIER_PATH
)

with torch.no_grad():

    _, sparse_features = model(
        features
    )

sparse_features = sparse_features.cpu().numpy()

probabilities = classifier.predict_proba(
    sparse_features
)[:, 1]

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

print(
    "Threshold Analysis"
)

print(
    "=================="
)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        labels,
        predictions
    )

    true_negative = matrix[0][0]
    false_positive = matrix[0][1]

    false_positive_rate = (
        false_positive /
        (true_negative + false_positive)
    )

    print(
        f"\nThreshold: {threshold:.2f}"
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