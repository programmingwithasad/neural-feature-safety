import torch
import numpy as np
import pandas as pd
import joblib

from transformers import AutoTokenizer, AutoModel
from autoencoder import SparseAutoencoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


MODEL_NAME = "distilbert-base-uncased"
SAE_PATH = "models/sparse_autoencoder_full.pt"
CLASSIFIER_PATH = "models/conversation_enhanced_safety_classifier.pkl"
DATA_PATH = "data/processed/safety_challenge_set.csv"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    f"Using device: {device}"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(device)
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
        sae_data.get("state_dict", sae_data)
    )
else:
    state_dict = sae_data

sae.load_state_dict(
    state_dict
)

sae = sae.to(device)
sae.eval()


classifier = joblib.load(
    CLASSIFIER_PATH
)


df = pd.read_csv(
    DATA_PATH
)


def extract_features(text):

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
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
        ).sum(dim=1) / mask.sum(
            dim=1
        ).clamp(min=1)

        _, features = sae(
            pooled
        )

    return features.squeeze(
        0
    ).cpu().numpy()


features = []

for i, text in enumerate(
    df["text"]
):

    features.append(
        extract_features(text)
    )

    if (
        (i + 1) % 25 == 0
        or i + 1 == len(df)
    ):
        print(
            f"Processed: {i + 1} / {len(df)}"
        )


X = np.asarray(
    features
)

probabilities = classifier.predict_proba(
    X
)[:, 1]


df["harmful_probability"] = probabilities


for threshold in [
    0.50,
    0.60,
    0.70
]:

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    y_true = (
        df["label"] == "harmful"
    ).astype(np.int64)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        predictions
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"Threshold: {threshold:.2f}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Harmful Precision: {precision:.4f}"
    )

    print(
        f"Harmful Recall: {recall:.4f}"
    )

    print(
        f"Harmful F1: {f1:.4f}"
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        matrix
    )


print()
print(
    "=" * 70
)

print(
    "BENIGN FALSE POSITIVES AT THRESHOLD 0.60"
)

benign = df[
    df["label"] == "unharmful"
].copy()

benign_fp = benign[
    benign[
        "harmful_probability"
    ] >= 0.60
]

print(
    f"False positives: {len(benign_fp)} / {len(benign)}"
)

print(
    f"False positive rate: {len(benign_fp) / len(benign):.4f}"
)

print()

for _, row in benign_fp.sort_values(
    "harmful_probability",
    ascending=False
).iterrows():

    print(
        f"{row['harmful_probability']:.4f} | "
        f"{row['category']} | "
        f"{row['text']}"
    )


print()
print(
    "=" * 70
)

print(
    "CATEGORY ANALYSIS AT THRESHOLD 0.60"
)

df["prediction"] = (
    df["harmful_probability"] >= 0.60
).astype(np.int64)

df["true_label"] = (
    df["label"] == "harmful"
).astype(np.int64)


for category, group in df.groupby(
    "category"
):

    y_true_category = group[
        "true_label"
    ]

    y_pred_category = group[
        "prediction"
    ]

    accuracy = accuracy_score(
        y_true_category,
        y_pred_category
    )

    print(
        f"{category:25s} "
        f"n={len(group):3d} "
        f"accuracy={accuracy:.4f} "
        f"mean_probability={group['harmful_probability'].mean():.4f}"
    )


OUTPUT_PATH = (
    "data/processed/"
    "safety_challenge_results.csv"
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print(
    f"Saved to: {OUTPUT_PATH}"
)