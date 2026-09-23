import torch
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import SparseAutoencoder, sparse_loss


DATA_PATH = "data/processed/prompt_activations_10000.pt"
MODEL_PATH = "models/sparse_autoencoder_10000.pt"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

data = torch.load(
    DATA_PATH,
    map_location=device
)

features = data["features"].float()

dataset = TensorDataset(features)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = SparseAutoencoder(
    input_dim=768,
    hidden_dim=2048
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(EPOCHS):

    total_loss = 0.0

    for batch in loader:

        x = batch[0].to(device)

        optimizer.zero_grad()

        reconstructed, sparse_features = model(x)

        loss = sparse_loss(
            x,
            reconstructed,
            sparse_features
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        "Epoch:",
        epoch + 1,
        "/",
        EPOCHS,
        "Loss:",
        average_loss
    )

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\nTraining completed.")

print("Model saved to:", MODEL_PATH)