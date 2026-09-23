import torch

from autoencoder import SparseAutoencoder


DATA_PATH = "data/processed/prompt_activations_1000.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = torch.load(
    DATA_PATH,
    map_location=device
)

features = data["features"]

model = SparseAutoencoder(
    input_dim=768,
    hidden_dim=2048
).to(device)

features = features.to(device)

reconstructed, sparse_features = model(
    features
)

print("Input shape:", features.shape)

print("Sparse feature shape:", sparse_features.shape)

print("Reconstructed shape:", reconstructed.shape)

print("Device:", features.device)