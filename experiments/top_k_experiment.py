import torch
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from autoencoder import SparseAutoencoder


TRAIN_DATA_PATH = "data/processed/prompt_activations_10000.pt"
VALIDATION_DATA_PATH = "data/processed/prompt_validation_activations.pt"
SAE_MODEL_PATH = "models/sparse_autoencoder_10000.pt"
ANALYSIS_PATH = "data/processed/sae_feature_analysis.csv"

TOP_K_VALUES = [10, 25, 50, 100, 250, 500, 2048]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_data = torch.load(
    TRAIN_DATA_PATH,
    map_location=device
)

validation_data = torch.load(
    VALIDATION_DATA_PATH,
    map_location=device
)

train_features = train_data["features"].float().to(device)
train_labels = train_data["labels"].cpu().numpy()

validation_features = validation_data["features"].float().to(device)
validation_labels = validation_data["labels"].cpu().numpy()

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

    _, validation_sparse = sae(
        validation_features
    )

train_sparse = train_sparse.cpu().numpy()
validation_sparse = validation_sparse.cpu().numpy()

analysis = pd.read_csv(
    ANALYSIS_PATH
)

ranked_features = analysis[
    "feature"
].astype(int).tolist()

results = []

for k in TOP_K_VALUES:

    selected_features = ranked_features[:k]

    X_train = train_sparse[
        :,
        selected_features
    ]

    X_validation = validation_sparse[
        :,
        selected_features
    ]

    classifier = LogisticRegression(
        max_iter=1000
    )

    classifier.fit(
        X_train,
        train_labels
    )

    predictions = classifier.predict(
        X_validation
    )

    accuracy = accuracy_score(
        validation_labels,
        predictions
    )

    precision = precision_score(
        validation_labels,
        predictions,
        pos_label=1
    )

    recall = recall_score(
        validation_labels,
        predictions,
        pos_label=1
    )

    f1 = f1_score(
        validation_labels,
        predictions,
        pos_label=1
    )

    results.append({
        "top_k": k,
        "accuracy": accuracy,
        "harmful_precision": precision,
        "harmful_recall": recall,
        "harmful_f1": f1
    })

    print(
        f"Top {k}: "
        f"Accuracy={accuracy:.4f}, "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1={f1:.4f}"
    )

results_df = pd.DataFrame(
    results
)

print("\nComplete Results:")
print(
    results_df.to_string(
        index=False
    )
)

best_result = results_df.loc[
    results_df["harmful_f1"].idxmax()
]

print("\nBest Top-K based on harmful F1:")

print(
    best_result.to_string()
)