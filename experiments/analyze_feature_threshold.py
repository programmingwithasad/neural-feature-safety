import torch
import numpy as np

from autoencoder import SparseAutoencoder


DATA_PATH = "data/processed/prompt_train_activations_full.pt"
MODEL_PATH = "models/sparse_autoencoder_full.pt"

FEATURE_INDEX = 1273


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = torch.load(
    DATA_PATH,
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

activation = sparse_features[
    :,
    FEATURE_INDEX
].cpu().numpy()

labels = labels.numpy()

harmful_activation = activation[
    labels == 1
]

unharmful_activation = activation[
    labels == 0
]

print(
    "Feature:",
    FEATURE_INDEX
)

print(
    "Harmful samples:",
    len(harmful_activation)
)

print(
    "Unharmful samples:",
    len(unharmful_activation)
)

print(
    "\nHarmful activation statistics:"
)

print(
    "Mean:",
    harmful_activation.mean()
)

print(
    "Median:",
    np.median(harmful_activation)
)

print(
    "Maximum:",
    harmful_activation.max()
)

print(
    "\nUnharmful activation statistics:"
)

print(
    "Mean:",
    unharmful_activation.mean()
)

print(
    "Median:",
    np.median(unharmful_activation)
)

print(
    "Maximum:",
    unharmful_activation.max()
)

thresholds = [
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5
]

print(
    "\nThreshold analysis:"
)

for threshold in thresholds:

    harmful_rate = (
        harmful_activation >= threshold
    ).mean() * 100

    unharmful_rate = (
        unharmful_activation >= threshold
    ).mean() * 100

    print(
        f"Threshold {threshold:.2f}: "
        f"Harmful={harmful_rate:.2f}% "
        f"Unharmful={unharmful_rate:.2f}%"
    )
    