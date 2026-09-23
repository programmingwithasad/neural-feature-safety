import torch

from autoencoder import SparseAutoencoder


DATA_PATH = "data/processed/prompt_activations_1000.pt"
MODEL_PATH = "models/sparse_autoencoder.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = torch.load(
    DATA_PATH,
    map_location=device
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

    reconstruction_error = torch.mean(
        (features - reconstructed) ** 2
    )

    threshold = 0.01

    active_features = (
        sparse_features > threshold
    ).float()

    average_active_features = (
        active_features.sum(dim=1).mean()
    )

    total_features = sparse_features.shape[1]

    sparsity_percentage = (
        1
        - average_active_features / total_features
    ) * 100

print("Reconstruction MSE:", reconstruction_error.item())

print(
    "Average active features:",
    average_active_features.item()
)

print(
    "Total latent features:",
    total_features
)

print(
    "Sparsity percentage:",
    sparsity_percentage.item()
)