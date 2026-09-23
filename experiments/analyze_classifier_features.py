import torch
import pandas as pd

from sklearn.linear_model import LogisticRegression

from autoencoder import SparseAutoencoder


TRAIN_DATA_PATH = "data/processed/prompt_train_activations_full.pt"
SAE_MODEL_PATH = "models/sparse_autoencoder_full.pt"
OUTPUT_PATH = "data/processed/classifier_feature_weights.csv"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = torch.load(
    TRAIN_DATA_PATH,
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
        SAE_MODEL_PATH,
        map_location=device
    )
)

model.eval()

with torch.no_grad():

    _, sparse_features = model(
        features
    )

sparse_features = sparse_features.cpu().numpy()

classifier = LogisticRegression(
    max_iter=1000
)

classifier.fit(
    sparse_features,
    labels
)

weights = classifier.coef_[0]

analysis = pd.DataFrame({
    "feature": range(2048),
    "weight": weights,
    "absolute_weight": abs(weights)
})

analysis = analysis.sort_values(
    "absolute_weight",
    ascending=False
)

print(
    "Top 20 classifier-associated features:"
)

print(
    analysis.head(20).to_string(
        index=False
    )
)

print(
    "\nMost harmful-associated classifier features:"
)

positive = analysis[
    analysis["weight"] > 0
].sort_values(
    "weight",
    ascending=False
)

print(
    positive.head(10).to_string(
        index=False
    )
)

print(
    "\nMost unharmful-associated classifier features:"
)

negative = analysis[
    analysis["weight"] < 0
].sort_values(
    "weight"
)

print(
    negative.head(10).to_string(
        index=False
    )
)

analysis.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "\nSaved to:",
    OUTPUT_PATH
)