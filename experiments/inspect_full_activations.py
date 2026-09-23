import torch


DATA_PATH = "data/processed/prompt_train_activations_full.pt"


data = torch.load(
    DATA_PATH,
    map_location="cpu"
)

features = data["features"].float()
labels = data["labels"]

print("Feature shape:", features.shape)
print("Label shape:", labels.shape)

print("\nHarmful samples:", (labels == 1).sum().item())
print("Unharmful samples:", (labels == 0).sum().item())

print("\nNaN values:", torch.isnan(features).sum().item())
print("Infinite values:", torch.isinf(features).sum().item())

print("\nMinimum feature value:", features.min().item())
print("Maximum feature value:", features.max().item())
print("Mean feature value:", features.mean().item())
print("Feature standard deviation:", features.std().item())