import torch

from autoencoder import SparseAutoencoder


INPUT_PATH = "data/processed/prompt_validation_activations.pt"
MODEL_PATH = "models/sparse_autoencoder_full.pt"
OUTPUT_PATH = "data/processed/prompt_validation_sae_features.pt"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = torch.load(
    INPUT_PATH,
    map_location="cpu"
)

features = data["features"].float().to(device)
labels = data["labels"]

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

output = {
    "features": sparse_features.cpu(),
    "labels": labels
}

torch.save(
    output,
    OUTPUT_PATH
)

print(
    "Validation SAE features:",
    sparse_features.shape
)

print(
    "Labels:",
    labels.shape
)

print(
    "Saved to:",
    OUTPUT_PATH
)