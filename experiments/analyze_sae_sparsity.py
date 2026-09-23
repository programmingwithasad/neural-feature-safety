import torch
import numpy as np


FILES = [
    "data/processed/prompt_train_sae_features_full.pt",
    "data/processed/oasst_benign_sae_features.pt",
    "data/processed/oasst_hard_negative_sae_features.pt",
    "data/processed/prompt_validation_sae_features.pt",
    "data/processed/prompt_test_sae_features.pt"
]


def load_features(path):
    data = torch.load(
        path,
        map_location="cpu"
    )

    features = data["features"]

    if torch.is_tensor(features):
        features = features.numpy()

    return features


for path in FILES:

    X = load_features(path)

    active = (
        X > 0
    ).sum(axis=1)

    positive = X[
        X > 0
    ]

    print()
    print("=" * 70)
    print(path)
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")

    print()
    print("Active features per sample")

    print(
        f"Mean: {active.mean():.2f}"
    )

    print(
        f"Median: {np.median(active):.2f}"
    )

    print(
        f"Minimum: {active.min()}"
    )

    print(
        f"Maximum: {active.max()}"
    )

    print(
        f"25th percentile: {np.percentile(active, 25):.2f}"
    )

    print(
        f"75th percentile: {np.percentile(active, 75):.2f}"
    )

    print(
        f"90th percentile: {np.percentile(active, 90):.2f}"
    )

    print()

    if len(positive) > 0:

        print(
            f"Mean positive activation: {positive.mean():.6f}"
        )

        print(
            f"Median positive activation: {np.median(positive):.6f}"
        )

        print(
            f"Maximum activation: {positive.max():.6f}"
        )

    print(
        f"Average sparsity: {1 - active.mean() / X.shape[1]:.4%}"
    )