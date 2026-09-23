import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    ROOT_DIR,
    "src"
)

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
    "candidate_challenge_comparison.csv"
)

CANDIDATES = [
    (318, 0.20),
    (146, 0.20),
    (1049, 0.05),
    (1049, 0.20),
    (850, 0.20)
]

THRESHOLD = 0.60
BATCH_SIZE = 32


def load_models():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
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

    batches = []

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

            batches.append(
                sparse_features.cpu()
            )

        print(
            f"Processed {min(start + BATCH_SIZE, len(texts))}/{len(texts)}"
        )

    return torch.cat(
        batches,
        dim=0
    )


def calculate_metrics(
    labels,
    probabilities
):

    predictions = (
        probabilities >= THRESHOLD
    )

    tp = int(
        np.sum(
            (labels == 1) & predictions
        )
    )

    tn = int(
        np.sum(
            (labels == 0) & ~predictions
        )
    )

    fp = int(
        np.sum(
            (labels == 0) & predictions
        )
    )

    fn = int(
        np.sum(
            (labels == 1) & ~predictions
        )
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
    print("CANDIDATE FEATURE CHALLENGE COMPARISON")
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

    texts = (
        df["text"]
        .astype(str)
        .tolist()
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

    baseline = calculate_metrics(
        labels,
        baseline_probability
    )

    print("\nBaseline")

    print(
        baseline
    )

    results = []

    for feature, suppression in CANDIDATES:

        modified_features = features.clone()

        modified_features[
            :,
            feature
        ] *= (
            1.0 - suppression
        )

        probability = (
            classifier.predict_proba(
                modified_features.numpy()
            )[:, 1]
        )

        metrics = calculate_metrics(
            labels,
            probability
        )

        row = {
            "feature": feature,
            "suppression_rate": suppression,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "fp_change":
                metrics["fp"] - baseline["fp"],
            "fn_change":
                metrics["fn"] - baseline["fn"],
            "tp_change":
                metrics["tp"] - baseline["tp"],
            "tn_change":
                metrics["tn"] - baseline["tn"]
        }

        results.append(
            row
        )

        print(
            f"\nFeature={feature} "
            f"Suppression={suppression:.2f}"
        )

        print(
            f"Accuracy={metrics['accuracy']:.4f}"
        )

        print(
            f"Precision={metrics['precision']:.4f}"
        )

        print(
            f"Recall={metrics['recall']:.4f}"
        )

        print(
            f"F1={metrics['f1']:.4f}"
        )

        print(
            f"TP={metrics['tp']} "
            f"TN={metrics['tn']} "
            f"FP={metrics['fp']} "
            f"FN={metrics['fn']}"
        )

        print(
            f"FP change={row['fp_change']} "
            f"FN change={row['fn_change']}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nComparison")

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()