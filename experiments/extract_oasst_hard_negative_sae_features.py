import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from autoencoder import SparseAutoencoder


INPUT_CSV = "data/processed/oasst_curated_hard_negatives.csv"
OUTPUT_PATH = "data/processed/oasst_hard_negative_sae_features.pt"
SAE_PATH = "models/sparse_autoencoder_full.pt"

MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 32
MAX_LENGTH = 128


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ---------------------------------------------------------
# Load hard-negative data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

texts = df["text"].tolist()
labels = df["label"].tolist()

print(f"Hard-negative examples: {len(texts)}")


# ---------------------------------------------------------
# Load DistilBERT
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()


# ---------------------------------------------------------
# Load existing full-data SAE
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Extract DistilBERT activations
# ---------------------------------------------------------

all_features = []

with torch.no_grad():

    for start in range(
        0,
        len(texts),
        BATCH_SIZE
    ):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        outputs = model(**encoded)

        hidden = outputs.last_hidden_state

        mask = (
            encoded["attention_mask"]
            .unsqueeze(-1)
        )

        pooled = (
            hidden * mask
        ).sum(dim=1) / mask.sum(
            dim=1
        ).clamp(min=1)

        all_features.append(
            pooled.cpu()
        )

        processed = min(
            start + len(batch_texts),
            len(texts)
        )

        print(
            f"Processed: {processed} / {len(texts)}"
        )


features = torch.cat(
    all_features
)


# ---------------------------------------------------------
# SAE encoding
# ---------------------------------------------------------

with torch.no_grad():

    features = features.to(device)

    # The existing SAE uses:
    # encoder -> ReLU -> sparse features

    sparse_features = torch.relu(
        sae.encoder(features)
    )

    sparse_features = (
        sparse_features.cpu()
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output = {
    "features": sparse_features,
    "labels": labels,
    "texts": texts
}

torch.save(
    output,
    OUTPUT_PATH
)

print()
print("Hard-negative SAE extraction completed.")
print(
    f"SAE feature shape: {sparse_features.shape}"
)
print(
    f"Labels: {len(labels)}"
)
print(
    f"Saved to: {OUTPUT_PATH}"
)