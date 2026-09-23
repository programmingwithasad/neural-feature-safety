import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


TRAIN_PATH = "data/processed/prompt_train_activations_full.pt"
VALIDATION_PATH = "data/processed/prompt_validation_activations.pt"
TEST_PATH = "data/processed/prompt_test_activations.pt"
SAE_PATH = "models/sparse_autoencoder_full.pt"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_data = torch.load(
    TRAIN_PATH,
    map_location="cpu"
)

validation_data = torch.load(
    VALIDATION_PATH,
    map_location="cpu"
)

test_data = torch.load(
    TEST_PATH,
    map_location="cpu"
)

from autoencoder import SparseAutoencoder

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

train_inputs = train_data["features"].float().to(device)
validation_inputs = validation_data["features"].float().to(device)
test_inputs = test_data["features"].float().to(device)

train_labels = train_data["labels"].numpy()
validation_labels = validation_data["labels"].numpy()
test_labels = test_data["labels"].numpy()

with torch.no_grad():

    _, train_sae = model(
        train_inputs
    )

    _, validation_sae = model(
        validation_inputs
    )

    _, test_sae = model(
        test_inputs
    )

train_sae = train_sae.cpu().numpy()
validation_sae = validation_sae.cpu().numpy()
test_sae = test_sae.cpu().numpy()

feature_scores = torch.tensor(
    train_sae
).std(
    dim=0
).numpy()

ranked_features = feature_scores.argsort()[::-1]

top_k_values = [
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    1500,
    2048
]

results = []

print(
    "Running Top-K experiment..."
)

for k in top_k_values:

    selected = ranked_features[:k]

    classifier = LogisticRegression(
        max_iter=1000
    )

    classifier.fit(
        train_sae[:, selected],
        train_labels
    )

    validation_predictions = classifier.predict(
        validation_sae[:, selected]
    )

    accuracy = accuracy_score(
        validation_labels,
        validation_predictions
    )

    precision = precision_score(
        validation_labels,
        validation_predictions,
        pos_label=1
    )

    recall = recall_score(
        validation_labels,
        validation_predictions,
        pos_label=1
    )

    f1 = f1_score(
        validation_labels,
        validation_predictions,
        pos_label=1
    )

    results.append(
        (
            k,
            accuracy,
            precision,
            recall,
            f1
        )
    )

    print(
        f"Top {k}: "
        f"Accuracy={accuracy:.4f}, "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1={f1:.4f}"
    )

best = max(
    results,
    key=lambda x: x[4]
)

best_k = best[0]

print(
    "\nBest Top-K:",
    best_k
)

print(
    "Validation Accuracy:",
    best[1]
)

print(
    "Validation Harmful Precision:",
    best[2]
)

print(
    "Validation Harmful Recall:",
    best[3]
)

print(
    "Validation Harmful F1:",
    best[4]
)

selected = ranked_features[:best_k]

final_classifier = LogisticRegression(
    max_iter=1000
)

final_classifier.fit(
    train_sae[:, selected],
    train_labels
)

test_predictions = final_classifier.predict(
    test_sae[:, selected]
)

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

test_precision = precision_score(
    test_labels,
    test_predictions,
    pos_label=1
)

test_recall = recall_score(
    test_labels,
    test_predictions,
    pos_label=1
)

test_f1 = f1_score(
    test_labels,
    test_predictions,
    pos_label=1
)

print(
    "\nFinal Test Results"
)

print(
    "Top-K:",
    best_k
)

print(
    "Accuracy:",
    test_accuracy
)

print(
    "Harmful Precision:",
    test_precision
)

print(
    "Harmful Recall:",
    test_recall
)

print(
    "Harmful F1:",
    test_f1
)