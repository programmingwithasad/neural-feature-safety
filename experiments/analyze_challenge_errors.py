import torch
import numpy as np
import pandas as pd
import joblib

from transformers import AutoTokenizer, AutoModel
from autoencoder import SparseAutoencoder


MODEL_NAME = "distilbert-base-uncased"
SAE_PATH = "models/sparse_autoencoder_full.pt"
CLASSIFIER_PATH = "models/conversation_enhanced_safety_classifier.pkl"
RESULTS_PATH = "data/processed/safety_challenge_results.csv"

THRESHOLD = 0.60


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
    RESULTS_PATH
)


df["true_label"] = (
    df["label"] == "harmful"
).astype(int)

df["prediction"] = (
    df["harmful_probability"] >= THRESHOLD
).astype(int)


false_positives = df[
    (df["true_label"] == 0) &
    (df["prediction"] == 1)
].copy()

false_negatives = df[
    (df["true_label"] == 1) &
    (df["prediction"] == 0)
].copy()


print()
print("=" * 70)
print("CHALLENGE SET ERROR ANALYSIS")
print("=" * 70)

print()
print(
    f"False positives: {len(false_positives)}"
)

print(
    f"False negatives: {len(false_negatives)}"
)

print()
print("FALSE POSITIVES")
print("=" * 70)

for _, row in false_positives.sort_values(
    "harmful_probability",
    ascending=False
).iterrows():

    print(
        f"{row['harmful_probability']:.4f} | "
        f"{row['category']} | "
        f"{row['text']}"
    )


print()
print("FALSE NEGATIVES")
print("=" * 70)

for _, row in false_negatives.sort_values(
    "harmful_probability"
).iterrows():

    print(
        f"{row['harmful_probability']:.4f} | "
        f"{row['category']} | "
        f"{row['text']}"
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


weights = classifier.named_steps[
    "classifier"
].coef_[0]


def analyze_example(
    text,
    probability
):

    features = extract_features(
        text
    )

    contributions = (
        features * weights
    )

    positive_indices = np.argsort(
        contributions
    )[::-1][:20]

    negative_indices = np.argsort(
        contributions
    )[:20]

    print()
    print("=" * 70)

    print(
        f"Text: {text}"
    )

    print(
        f"Harmful probability: {probability:.4f}"
    )

    print()
    print(
        "Top harmful feature contributions"
    )

    for index in positive_indices:

        print(
            f"Feature {index:4d} | "
            f"activation={features[index]:.6f} | "
            f"weight={weights[index]:.6f} | "
            f"contribution={contributions[index]:.6f}"
        )

    print()
    print(
        "Top unharmful feature contributions"
    )

    for index in negative_indices:

        print(
            f"Feature {index:4d} | "
            f"activation={features[index]:.6f} | "
            f"weight={weights[index]:.6f} | "
            f"contribution={contributions[index]:.6f}"
        )


print()
print("=" * 70)
print("FALSE POSITIVE FEATURE ANALYSIS")
print("=" * 70)

for _, row in false_positives.iterrows():

    analyze_example(
        row["text"],
        row["harmful_probability"]
    )


print()
print("=" * 70)
print("FALSE NEGATIVE FEATURE ANALYSIS")
print("=" * 70)

for _, row in false_negatives.iterrows():

    analyze_example(
        row["text"],
        row["harmful_probability"]
    )


output_rows = []

for _, row in df.iterrows():

    features = extract_features(
        row["text"]
    )

    contributions = (
        features * weights
    )

    top_features = np.argsort(
        contributions
    )[::-1][:20]

    output_rows.append(
        {
            "text": row["text"],
            "category": row["category"],
            "label": row["label"],
            "harmful_probability": row[
                "harmful_probability"
            ],
            "prediction": row[
                "prediction"
            ],
            "top_features": ",".join(
                map(
                    str,
                    top_features
                )
            )
        }
    )


output = pd.DataFrame(
    output_rows
)

output_path = (
    "data/processed/"
    "challenge_feature_error_analysis.csv"
)

output.to_csv(
    output_path,
    index=False
)

print()
print(
    f"Saved to: {output_path}"
)
