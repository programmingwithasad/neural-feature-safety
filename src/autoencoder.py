import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=2048):
        super().__init__()

        self.encoder = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.decoder = nn.Linear(
            hidden_dim,
            input_dim
        )

    def forward(self, x):

        encoded = self.encoder(x)

        sparse_features = torch.relu(encoded)

        reconstructed = self.decoder(
            sparse_features
        )

        return reconstructed, sparse_features


def sparse_loss(
    original,
    reconstructed,
    sparse_features,
    sparsity_weight=0.001
):

    reconstruction_loss = torch.mean(
        (original - reconstructed) ** 2
    )

    sparsity_loss = torch.mean(
        torch.abs(sparse_features)
    )

    total_loss = (
        reconstruction_loss
        + sparsity_weight * sparsity_loss
    )

    return total_loss