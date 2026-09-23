import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.insert(0, SRC_DIR)

from autoencoder import SparseAutoencoder

CHALLENGE_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "safety_challenge_set.csv"
)

SAE_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "sparse_autoencoder_full.pt"
)

CLASSIFIER_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "final_safety_classifier.pkl"
)

OUTPUT_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "feature_850_challenge_results.csv"
)

FEATURE_INDEX = 850
SUPPRESSION_RATE = 0.20
CLASSIFIER_THRESHOLD = 0.60
BATCH_SIZE = 32


def load_models():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    distilbert = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    ).to(device)

    distilbert.eval()

    sae = SparseAutoencoder(
        input_dim=768,
        hidden_dim=2048
    ).to(device)

    sae.load_state_dict(
        torch.load(
            SAE_PATH,
            map_location=device,
            weights_only=False
        )
    )

    sae.eval()

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    return (
        tokenizer,
        distilbert,
        sae,
        classifier,
        device
    )


def extract_features(
    texts,
    tokenizer,
    distilbert,
    sae,
    device
):

    all_features = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE
    ):

        batch = texts[
            start:start + BATCH_SIZE
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

            all_features.append(
                sparse_features.cpu()
            )

        print(
            f"Processed {min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)}"
        )

    return torch.cat(
        all_features,
        dim=0
    )


def calculate_metrics(
    labels,
    probabilities
):

    predictions = (
        probabilities >= CLASSIFIER_THRESHOLD
    )

    tp = np.sum(
        (labels == 1) & predictions
    )

    tn = np.sum(
        (labels == 0) & ~predictions
    )

    fp = np.sum(
        (labels == 0) & predictions
    )

    fn = np.sum(
        (labels == 1) & ~predictions
    )

    accuracy = (
        tp + tn
    ) / len(labels)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def main():

    print("=" * 60)
    print("FEATURE 850 SAFETY CHALLENGE EVALUATION")
    print("=" * 60)

    df = pd.read_csv(
        CHALLENGE_PATH
    )

    labels = (
        df["label"]
        .map(
            {
                "unharmful": 0,
                "harmful": 1
            }
        )
        .values
    )

    texts = df[
        "text"
    ].astype(
        str
    ).tolist()

    print(
        f"Challenge samples: {len(texts)}"
    )

    (
        tokenizer,
        distilbert,
        sae,
        classifier,
        device
    ) = load_models()

    print(
        f"Device: {device}"
    )

    features = extract_features(
        texts,
        tokenizer,
        distilbert,
        sae,
        device
    )

    baseline_probability = (
        classifier.predict_proba(
            features.numpy()
        )[:, 1]
    )

    modified_features = features.clone()

    modified_features[
        :,
        FEATURE_INDEX
    ] *= (
        1.0 - SUPPRESSION_RATE
    )

    intervention_probability = (
        classifier.predict_proba(
            modified_features.numpy()
        )[:, 1]
    )

    baseline_metrics = calculate_metrics(
        labels,
        baseline_probability
    )

    intervention_metrics = calculate_metrics(
        labels,
        intervention_probability
    )

    print("\nBaseline")

    for key, value in baseline_metrics.items():

        print(
            f"{key}: {value}"
        )

    print("\nFeature 850 Intervention")

    for key, value in intervention_metrics.items():

        print(
            f"{key}: {value}"
        )

    print("\nChanges")

    print(
        "False positive change:",
        intervention_metrics["fp"]
        - baseline_metrics["fp"]
    )

    print(
        "False negative change:",
        intervention_metrics["fn"]
        - baseline_metrics["fn"]
    )

    print(
        "True positive change:",
        intervention_metrics["tp"]
        - baseline_metrics["tp"]
    )

    print(
        "True negative change:",
        intervention_metrics["tn"]
        - baseline_metrics["tn"]
    )

    results = df.copy()

    results[
        "baseline_probability"
    ] = baseline_probability

    results[
        "intervention_probability"
    ] = intervention_probability

    results[
        "probability_change"
    ] = (
        intervention_probability
        - baseline_probability
    )

    results[
        "baseline_decision"
    ] = np.where(
        baseline_probability
        >= CLASSIFIER_THRESHOLD,
        "BLOCK",
        "ALLOW"
    )

    results[
        "intervention_decision"
    ] = np.where(
        intervention_probability
        >= CLASSIFIER_THRESHOLD,
        "BLOCK",
        "ALLOW"
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()