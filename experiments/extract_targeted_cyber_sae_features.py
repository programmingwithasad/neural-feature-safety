import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from autoencoder import SparseAutoencoder


INPUT_PATH = "data/processed/targeted_cyber_dataset.csv"
SAE_PATH = "models/sparse_autoencoder_full.pt"
OUTPUT_PATH = "data/processed/targeted_cyber_sae_features.pt"

MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 32

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    f"Using device: {device}"
)

df = pd.read_csv(
    INPUT_PATH
)

texts = (
    df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)

labels = (
    df["label"]
    .map(
        {
            "unharmful": 0,
            "harmful": 1
        }
    )
    .astype(int)
    .tolist()
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

sae_data = torch.load(
    SAE_PATH,
    map_location=device
)

if isinstance(sae_data, dict):
    state_dict = sae_data.get(
        "model_state_dict",
        sae_data.get(
            "state_dict",
            sae_data
        )
    )
else:
    state_dict = sae_data

sae.load_state_dict(
    state_dict
)

sae = sae.to(device)
sae.eval()

features = []

for start in range(
    0,
    len(texts),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(texts)
    )

    batch = texts[
        start:end
    ]

    encoded = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        output = model(
            **encoded
        )

        hidden = output.last_hidden_state

        mask = encoded[
            "attention_mask"
        ].unsqueeze(-1)

        pooled = (
            hidden * mask
        ).sum(dim=1)

        pooled = pooled / mask.sum(
            dim=1
        ).clamp(min=1)

        _, sparse_features = sae(
            pooled
        )

    features.append(
        sparse_features.cpu()
    )

    print(
        f"Processed: {end} / {len(texts)}"
    )

features = torch.cat(
    features,
    dim=0
)

labels = torch.tensor(
    labels,
    dtype=torch.long
)

torch.save(
    {
        "features": features,
        "labels": labels,
        "texts": texts,
        "categories": df[
            "target_category"
        ].tolist()
    },
    OUTPUT_PATH
)

print()
print(
    "Targeted cyber SAE extraction completed."
)

print(
    f"SAE feature shape: {features.shape}"
)

print(
    f"Labels shape: {labels.shape}"
)

print(
    f"Saved to: {OUTPUT_PATH}"
)
