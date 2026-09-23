import pandas as pd
import torch
import joblib
import numpy as np

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


MODEL_PATH = "models/targeted_safety_classifier.pkl"
CHALLENGE_PATH = "data/processed/safety_challenge_set.csv"
SAE_PATH = "models/sparse_autoencoder_full.pt"

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32
THRESHOLD = 0.60

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

df = pd.read_csv(
    CHALLENGE_PATH
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

from autoencoder import SparseAutoencoder

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

classifier = joblib.load(
    MODEL_PATH
)

texts = (
    df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)

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

    batch = texts[start:end]

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

        _, sparse = sae(
            pooled
        )

    features.append(
        sparse.cpu()
    )

features = torch.cat(
    features,
    dim=0
).numpy()

probabilities = classifier.predict_proba(
    features
)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)

y = df["label"].map(
    {
        "unharmful": 0,
        "harmful": 1
    }
).astype(int).to_numpy()

print()
print(
    "TARGETED CLASSIFIER CHALLENGE EVALUATION"
)

print(
    "=" * 70
)

print(
    f"Threshold: {THRESHOLD:.2f}"
)

print(
    f"Accuracy: {accuracy_score(y, predictions):.4f}"
)

print(
    f"Harmful Precision: {precision_score(y, predictions, zero_division=0):.4f}"
)

print(
    f"Harmful Recall: {recall_score(y, predictions, zero_division=0):.4f}"
)

print(
    f"Harmful F1: {f1_score(y, predictions, zero_division=0):.4f}"
)

print()

print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(y, predictions)
)

print()

for category in df["category"].unique():

    mask = df["category"].eq(category)

    category_accuracy = accuracy_score(
        y[mask],
        predictions[mask]
    )

    category_probability = probabilities[
        mask
    ].mean()

    print(
        f"{category:<25} "
        f"n={mask.sum():3d} "
        f"accuracy={category_accuracy:.4f} "
        f"mean_probability={category_probability:.4f}"
    )

print()

print(
    "INDIVIDUAL DECISIONS"
)

print(
    "=" * 70
)

for text, probability, prediction in zip(
    texts,
    probabilities,
    predictions
):

    decision = (
        "BLOCK"
        if prediction == 1
        else "ALLOW"
    )

    print(
        f"{probability:.4f} | "
        f"{decision:<5} | "
        f"{text}"
    )