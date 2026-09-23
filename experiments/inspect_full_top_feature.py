import sys
import torch

from autoencoder import SparseAutoencoder


DATA_PATH = "data/processed/prompt_train_activations_full.pt"
MODEL_PATH = "models/sparse_autoencoder_full.pt"

FEATURE_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 1273
TOP_N = 20


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

activation_values = sparse_features[
    :,
    FEATURE_INDEX
].cpu()

top_values, top_indices = torch.topk(
    activation_values,
    TOP_N
)

print(
    "Top examples for SAE feature:",
    FEATURE_INDEX
)

for rank in range(TOP_N):

    index = top_indices[rank].item()

    activation = top_values[rank].item()

    label = labels[index].item()

    label_name = (
        "harmful"
        if label == 1
        else "unharmful"
    )

    print(
        f"\nRank: {rank + 1}"
    )

    print(
        "Activation:",
        activation
    )

    print(
        "Label:",
        label_name
    )

    print(
        "Prompt:",
        texts[index]
    )