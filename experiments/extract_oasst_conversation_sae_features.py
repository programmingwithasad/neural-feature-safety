import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from autoencoder import SparseAutoencoder


INPUT_PATH = "data/processed/oasst_conversation_augmented.csv"
OUTPUT_PATH = "data/processed/oasst_conversation_sae_features.pt"
SAE_PATH = "models/sparse_autoencoder_full.pt"
MODEL_NAME = "distilbert-base-uncased"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

df = pd.read_csv(INPUT_PATH)

texts = df["text"].tolist()

labels = torch.tensor(
    (df["label"] == "harmful").astype(int).values,
    dtype=torch.long
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

sae = SparseAutoencoder(
    input_dim=768,
    hidden_dim=2048
)

sae.load_state_dict(
    torch.load(
        SAE_PATH,
        map_location=device
    )
)

sae.to(device)
sae.eval()

all_features = []

batch_size = 32

with torch.no_grad():

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch_texts = texts[
            start:start + batch_size
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        outputs = model(**encoded)

        hidden = outputs.last_hidden_state

        mask = encoded[
            "attention_mask"
        ].unsqueeze(-1)

        features = (
            hidden * mask
        ).sum(dim=1) / mask.sum(
            dim=1
        ).clamp(min=1)

        sparse_features = torch.relu(
            sae.encoder(features)
        )

        all_features.append(
            sparse_features.cpu()
        )

        print(
            f"Processed: {min(start + batch_size, len(texts))} / {len(texts)}"
        )

features = torch.cat(
    all_features,
    dim=0
)

torch.save(
    {
        "features": features,
        "labels": labels
    },
    OUTPUT_PATH
)

print()
print("Conversational SAE extraction completed.")
print(f"SAE feature shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Saved to: {OUTPUT_PATH}")