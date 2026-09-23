import torch
import pickle
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
from autoencoder import SparseAutoencoder


DATA_PATH = "data/processed/oasst_benign_train.csv"
SAE_PATH = "models/sparse_autoencoder_full.pt"
CLASSIFIER_PATH = "models/combined_safety_classifier.pkl"

OUTPUT_PATH = "data/processed/oasst_hard_negatives.csv"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

df = pd.read_csv(DATA_PATH)

texts = df["text"].fillna("").tolist()

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

encoder = AutoModel.from_pretrained(
    "distilbert-base-uncased"
).to(device)

encoder.eval()

sae = SparseAutoencoder(
    input_dim=768,
    hidden_dim=2048
).to(device)

sae.load_state_dict(
    torch.load(
        SAE_PATH,
        map_location=device
    )
)

sae.eval()

with open(
    CLASSIFIER_PATH,
    "rb"
) as file:
    classifier = pickle.load(file)

probabilities = []

batch_size = 32

for start in range(
    0,
    len(texts),
    batch_size
):

    batch = texts[
        start:start + batch_size
    ]

    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = encoder(
            **inputs
        )

        hidden_states = (
            outputs.last_hidden_state
        )

        attention_mask = (
            inputs["attention_mask"]
        )

        mask = attention_mask.unsqueeze(
            -1
        ).expand(
            hidden_states.size()
        ).float()

        masked_hidden_states = (
            hidden_states * mask
        )

        pooled = (
            masked_hidden_states.sum(
                dim=1
            )
            / mask.sum(
                dim=1
            )
        )

        _, sparse_features = sae(
            pooled
        )

        batch_probabilities = (
            classifier.predict_proba(
                sparse_features.cpu().numpy()
            )[:, 1]
        )

    probabilities.extend(
        batch_probabilities.tolist()
    )

df["harmful_probability"] = probabilities

df = df.sort_values(
    "harmful_probability",
    ascending=False
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "Hard-negative mining completed."
)

print(
    f"Total examples: {len(df)}"
)

print(
    f"Probability >= 0.90: "
    f"{(df['harmful_probability'] >= 0.90).sum()}"
)

print(
    f"Probability >= 0.80: "
    f"{(df['harmful_probability'] >= 0.80).sum()}"
)

print(
    f"Probability >= 0.70: "
    f"{(df['harmful_probability'] >= 0.70).sum()}"
)

print(
    f"Probability >= 0.60: "
    f"{(df['harmful_probability'] >= 0.60).sum()}"
)

print(
    f"Probability >= 0.50: "
    f"{(df['harmful_probability'] >= 0.50).sum()}"
)

print(
    f"Saved to: {OUTPUT_PATH}"
)