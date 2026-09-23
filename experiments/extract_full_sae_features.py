import torch

from autoencoder import SparseAutoencoder


INPUT_PATH = "data/processed/prompt_train_activations_full.pt"
MODEL_PATH = "models/sparse_autoencoder_full.pt"
OUTPUT_PATH = "data/processed/prompt_train_sae_features_full.pt"


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

sae_features_list = []

batch_size = 1024

for start in range(
    0,
    len(features),
    batch_size
):

    batch = features[
        start:start + batch_size
    ]

    with torch.no_grad():

        _, sae_features = model(
            batch
        )

    sae_features_list.append(
        sae_features.cpu()
    )

    processed = min(
        start + batch_size,
        len(features)
    )

    print(
        f"Processed: {processed} / {len(features)}"
    )

sae_features = torch.cat(
    sae_features_list
)

torch.save(
    {
        "features": sae_features,
        "labels": labels.cpu()
    },
    OUTPUT_PATH
)

print(
    "\nSAE feature extraction completed."
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