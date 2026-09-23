import torch
import joblib
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from autoencoder import SparseAutoencoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


MODEL_NAME = "distilbert-base-uncased"

SAE_PATH = "models/sparse_autoencoder_full.pt"

CLASSIFIER_PATH = (
    "models/final_conversation_weighted_classifier.pkl"
)

CHALLENGE_PATH = (
    "data/processed/safety_challenge_set.csv"
)

OUTPUT_PATH = (
    "data/processed/final_challenge_results.csv"
)

THRESHOLD = 0.50


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

distilbert = AutoModel.from_pretrained(
    MODEL_NAME
).to(device)

distilbert.eval()


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


classifier = joblib.load(
    CLASSIFIER_PATH
)


df = pd.read_csv(
    CHALLENGE_PATH
)


print(
    f"Challenge samples: {len(df)}"
)

print()


def get_features(texts):

    inputs = tokenizer(
        texts,
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

        outputs = distilbert(
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

        summed = masked_hidden_states.sum(
            dim=1
        )

        counts = mask.sum(
            dim=1
        )

        pooled = summed / counts

        _, sparse_features = sae(
            pooled
        )

    return sparse_features.detach().cpu().numpy()


texts = df["text"].astype(str).tolist()


all_features = []

batch_size = 32


for start in range(
    0,
    len(texts),
    batch_size
):

    end = min(
        start + batch_size,
        len(texts)
    )

    batch_features = get_features(
        texts[start:end]
    )

    all_features.append(
        batch_features
    )

    print(
        f"Processed: {end} / {len(texts)}"
    )


X = np.concatenate(
    all_features,
    axis=0
)


probabilities = classifier.predict_proba(
    X
)[:, 1]


predictions = (
    probabilities >= THRESHOLD
).astype(np.int64)


y_true = np.asarray(
    [
        1 if str(x).lower() == "harmful" else 0
        for x in df["label"]
    ],
    dtype=np.int64
)


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

tn, fp, fn, tp = matrix.ravel()


false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)

false_negative_rate = (
    fn / (fn + tp)
    if (fn + tp) > 0
    else 0
)


print()
print(
    "=" * 70
)

print(
    "FINAL MODEL CHALLENGE EVALUATION"
)

print(
    "=" * 70
)

print(
    f"Threshold: {THRESHOLD:.2f}"
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

print(
    f"False Positive Rate: "
    f"{false_positive_rate:.4f}"
)

print(
    f"False Negative Rate: "
    f"{false_negative_rate:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(
    matrix
)


df["harmful_probability"] = probabilities

df["prediction"] = predictions

df["decision"] = np.where(
    predictions == 1,
    "BLOCK",
    "ALLOW"
)


print()
print(
    "CATEGORY ANALYSIS"
)

print(
    "=" * 70
)


if "category" in df.columns:

    for category, group in df.groupby(
        "category"
    ):

        indices = group.index

        category_accuracy = accuracy_score(
            y_true[indices],
            predictions[indices]
        )

        category_probability = (
            probabilities[indices].mean()
        )

        print(
            f"{category:<25} "
            f"n={len(group):3d} "
            f"accuracy={category_accuracy:.4f} "
            f"mean_probability="
            f"{category_probability:.4f}"
        )


print()
print(
    "FALSE POSITIVES"
)

print(
    "=" * 70
)


false_positive_indices = np.where(
    (y_true == 0) &
    (predictions == 1)
)[0]


print(
    f"False positives: "
    f"{len(false_positive_indices)} / "
    f"{np.sum(y_true == 0)}"
)


for index in false_positive_indices:

    print(
        f"{probabilities[index]:.4f} | "
        f"{texts[index]}"
    )


print()
print(
    "FALSE NEGATIVES"
)

print(
    "=" * 70
)


false_negative_indices = np.where(
    (y_true == 1) &
    (predictions == 0)
)[0]


print(
    f"False negatives: "
    f"{len(false_negative_indices)} / "
    f"{np.sum(y_true == 1)}"
)


for index in false_negative_indices:

    print(
        f"{probabilities[index]:.4f} | "
        f"{texts[index]}"
    )


df.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print(
    f"Saved to: {OUTPUT_PATH}"
)
