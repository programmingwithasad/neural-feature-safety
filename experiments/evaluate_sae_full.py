import torch

from autoencoder import SparseAutoencoder


DATA_PATH = "data/processed/prompt_train_activations_full.pt"
MODEL_PATH = "models/sparse_autoencoder_full.pt"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = torch.load(
    DATA_PATH,
    map_location="cpu"
)

features = data["features"].float().to(device)

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

    reconstructed, sparse_features = model(
        features
    )

mse = torch.mean(
    (reconstructed - features) ** 2
).item()

active_threshold = 0.01

active_features = (
    sparse_features > active_threshold
).float().sum(dim=1)

average_active = active_features.mean().item()

total_features = sparse_features.shape[1]

active_percentage = (
    average_active / total_features
) * 100

sparsity_percentage = 100 - active_percentage

print(
    "Reconstruction MSE:",
    mse
)

print(
    "Average active features:",
    average_active
)

print(
    "Total latent features:",
    total_features
)

print(
    "Active feature percentage:",
    active_percentage
)

print(
    "Sparsity percentage:",
    sparsity_percentage
)
