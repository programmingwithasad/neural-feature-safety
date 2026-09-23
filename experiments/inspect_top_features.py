import torch
import pandas as pd

from autoencoder import SparseAutoencoder


DATA_PATH = "data/processed/prompt_activations_10000.pt"
MODEL_PATH = "models/sparse_autoencoder_10000.pt"
DATASET_PATH = "data/processed/prompt_train.csv"

FEATURE_ID = 1018
TOP_N = 20


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

activation_data = torch.load(
    DATA_PATH,
    map_location=device
)

features = activation_data["features"].float().to(device)
texts = activation_data["texts"]
labels = activation_data["labels"]

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

feature_values = sparse_features[
    :, FEATURE_ID
].cpu()

top_indices = torch.argsort(
    feature_values,
    descending=True
)[:TOP_N]

dataset = pd.read_csv(
    DATASET_PATH
)

print(
    "Top examples for SAE feature:",
    FEATURE_ID
)

print()

for rank, index in enumerate(
    top_indices.tolist(),
    start=1
):

    print("Rank:", rank)

    print(
        "Activation:",
        feature_values[index].item()
    )

    print(
        "Label:",
        "harmful" if labels[index].item() == 1 else "unharmful"
    )

    print(
        "Prompt:",
        texts[index]
    )

    print()