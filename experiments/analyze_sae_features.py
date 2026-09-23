import torch
import pandas as pd

from autoencoder import SparseAutoencoder


DATA_PATH = "data/processed/prompt_activations_10000.pt"
MODEL_PATH = "models/sparse_autoencoder_10000.pt"
OUTPUT_PATH = "data/processed/sae_feature_analysis.csv"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = torch.load(
    DATA_PATH,
    map_location=device
)

features = data["features"].float().to(device)
labels = data["labels"].to(device)

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

    _, sparse_features = model(features)

harmful_features = sparse_features[labels == 1]
unharmful_features = sparse_features[labels == 0]

harmful_mean = harmful_features.mean(dim=0)
unharmful_mean = unharmful_features.mean(dim=0)

difference = harmful_mean - unharmful_mean

results = pd.DataFrame({
    "feature": range(sparse_features.shape[1]),
    "harmful_mean": harmful_mean.cpu().numpy(),
    "unharmful_mean": unharmful_mean.cpu().numpy(),
    "difference": difference.cpu().numpy()
})

results["absolute_difference"] = (
    results["difference"].abs()
)

results = results.sort_values(
    "absolute_difference",
    ascending=False
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Total features analyzed:", len(results))

print("\nTop 20 safety-associated features:")

print(
    results.head(20).to_string(
        index=False
    )
)

print(
    "\nSaved to:",
    OUTPUT_PATH
)