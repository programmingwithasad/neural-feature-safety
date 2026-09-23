import torch
import joblib
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from autoencoder import SparseAutoencoder


MODEL_PATH = "models/targeted_safety_classifier.pkl"
SAE_PATH = "models/sparse_autoencoder_full.pt"
CHALLENGE_PATH = "data/processed/safety_challenge_set.csv"

MODEL_NAME = "distilbert-base-uncased"
THRESHOLD = 0.60
BATCH_SIZE = 32

FEATURES = [
    390,
    325,
    1298,
    1151,
    58,
    1744,
    1416,
    598,
    850,
    2028,
    1977
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

df = pd.read_csv(
    CHALLENGE_PATH
)

texts = (
    df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)

y = (
    df["label"]
    .map(
        {
            "unharmful": 0,
            "harmful": 1
        }
    )
    .astype(int)
    .to_numpy()
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

classifier = joblib.load(
    MODEL_PATH
)

feature_batches = []

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

    feature_batches.append(
        sparse.cpu()
    )

features = torch.cat(
    feature_batches,
    dim=0
).numpy()

scaler = classifier.named_steps[
    "scaler"
]

logistic = classifier.named_steps[
    "classifier"
]

scaled = scaler.transform(
    features
)

weights = logistic.coef_[0]

intercept = logistic.intercept_[0]

baseline_scores = (
    intercept
    + scaled @ weights
)

baseline_probabilities = (
    1.0
    / (
        1.0
        + np.exp(-baseline_scores)
    )
)

baseline_predictions = (
    baseline_probabilities >= THRESHOLD
).astype(int)

baseline_accuracy = accuracy_score(
    y,
    baseline_predictions
)

baseline_precision = precision_score(
    y,
    baseline_predictions,
    zero_division=0
)

baseline_recall = recall_score(
    y,
    baseline_predictions,
    zero_division=0
)

baseline_f1 = f1_score(
    y,
    baseline_predictions,
    zero_division=0
)

rows = []

for feature in FEATURES:

    modified = scaled.copy()

    modified[:, feature] = 0.0

    scores = (
        intercept
        + modified @ weights
    )

    probabilities = (
        1.0
        / (
            1.0
            + np.exp(-scores)
        )
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    benign_mask = y == 0

    harmful_mask = y == 1

    baseline_fp = (
        baseline_predictions[
            benign_mask
        ] == 1
    ).sum()

    ablated_fp = (
        predictions[
            benign_mask
        ] == 1
    ).sum()

    baseline_fn = (
        baseline_predictions[
            harmful_mask
        ] == 0
    ).sum()

    ablated_fn = (
        predictions[
            harmful_mask
        ] == 0
    ).sum()

    fp_reduction = (
        baseline_fp - ablated_fp
    )

    fn_increase = (
        ablated_fn - baseline_fn
    )

    rows.append(
        {
            "feature": feature,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": ablated_fp,
            "false_positive_reduction": fp_reduction,
            "false_negatives": ablated_fn,
            "false_negative_increase": fn_increase,
            "mean_probability_change": (
                probabilities.mean()
                - baseline_probabilities.mean()
            )
        }
    )

result = pd.DataFrame(
    rows
)

result = result.sort_values(
    [
        "false_positive_reduction",
        "false_negative_increase",
        "accuracy"
    ],
    ascending=[
        False,
        True,
        False
    ]
)

output_path = (
    "data/processed/"
    "feature_intervention_challenge_results.csv"
)

result.to_csv(
    output_path,
    index=False
)

print()
print(
    "SYSTEMATIC FEATURE INTERVENTION"
)

print(
    "=" * 70
)

print(
    f"Baseline accuracy: {baseline_accuracy:.4f}"
)

print(
    f"Baseline precision: {baseline_precision:.4f}"
)

print(
    f"Baseline recall: {baseline_recall:.4f}"
)

print(
    f"Baseline F1: {baseline_f1:.4f}"
)

print()

print(
    result.to_string(
        index=False
    )
)

print()

print(
    f"Saved to: {output_path}"
)