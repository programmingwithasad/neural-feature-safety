import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from autoencoder import SparseAutoencoder


TRAIN_PATH = "data/processed/prompt_train_activations_full.pt"
TEST_PATH = "data/processed/prompt_test_activations.pt"
SAE_PATH = "models/sparse_autoencoder_full.pt"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_data = torch.load(
    TRAIN_PATH,
    map_location="cpu"
)

test_data = torch.load(
    TEST_PATH,
    map_location="cpu"
)

train_inputs = train_data["features"].float().to(device)
train_labels = train_data["labels"].numpy()

test_inputs = test_data["features"].float().to(device)
test_labels = test_data["labels"].numpy()

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

with torch.no_grad():

    _, train_sparse = model(
        train_inputs
    )

    _, test_sparse = model(
        test_inputs
    )

train_sparse = train_sparse.cpu().numpy()
test_sparse = test_sparse.cpu().numpy()

print(
    "Training SAE features:",
    train_sparse.shape
)

print(
    "Testing SAE features:",
    test_sparse.shape
)

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