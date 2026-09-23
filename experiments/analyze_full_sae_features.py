import torch
import pandas as pd

from autoencoder import SparseAutoencoder


DATA_PATH = "data/processed/prompt_train_activations_full.pt"
MODEL_PATH = "models/sparse_autoencoder_full.pt"
OUTPUT_PATH = "data/processed/full_sae_feature_analysis.csv"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = torch.load(
    DATA_PATH,
    map_location="cpu"
)

features = data["features"].float().to(device)
labels = data["labels"]
texts = data["texts"]

model = SparseAutoencoder(
    input_dim=768,
    hidden_dim=2048
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

with torch.no_grad():

    _, sparse_features = model(
        features
    )

harmful_mask = labels == 1
unharmful_mask = labels == 0

harmful_features = sparse_features[
    harmful_mask.to(device)
]

unharmful_features = sparse_features[
    unharmful_mask.to(device)
]

harmful_mean = harmful_features.mean(
    dim=0
).cpu()

unharmful_mean = unharmful_features.mean(
    dim=0
).cpu()

difference = (
    harmful_mean -
    unharmful_mean
)

absolute_difference = difference.abs()

analysis = pd.DataFrame({
    "feature": range(2048),
    "harmful_mean": harmful_mean.numpy(),
    "unharmful_mean": unharmful_mean.numpy(),
    "difference": difference.numpy(),
    "absolute_difference": absolute_difference.numpy()
})

analysis = analysis.sort_values(
    "absolute_difference",
    ascending=False
)

analysis.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "Total features analyzed:",
    len(analysis)
)

print(
    "\nTop 20 safety-associated features:"
)

print(
    analysis.head(20).to_string(
        index=False
    )
)

print(
    "\nSaved to:",
    OUTPUT_PATH
)