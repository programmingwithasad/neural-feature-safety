import torch

from autoencoder import SparseAutoencoder


INPUT_PATH = "data/processed/oasst_benign_activations.pt"
MODEL_PATH = "models/sparse_autoencoder_full.pt"
OUTPUT_PATH = "data/processed/oasst_benign_sae_features.pt"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    "Using device:",
    device
)

data = torch.load(
    INPUT_PATH,
    map_location=device
)

features = data["features"]
labels = data["labels"]

print(
    "Input features:",
    features.shape
)

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

features = features.to(device)

with torch.no_grad():

    _, sae_features = model(
        features
    )

sae_features = sae_features.cpu()
labels = labels.cpu()

torch.save(
    {
        "features": sae_features,
        "labels": labels
    },
    OUTPUT_PATH
)

print(
    "\nSAE extraction completed."
)

print(
    "SAE feature shape:",
    sae_features.shape
)

print(
    "Labels shape:",
    labels.shape
)

print(
    "Saved to:",
    OUTPUT_PATH
)