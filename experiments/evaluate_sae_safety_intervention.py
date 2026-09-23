import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAE_FEATURES_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "prompt_train_sae_features_full.pt"
)

ANALYSIS_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "full_sae_feature_analysis.csv"
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
    "sae_safety_intervention_results.csv"
)

TOP_FEATURES = 20
SUPPRESSION_RATE = 1.0
THRESHOLD = 0.60


def load_data():

    sae_data = torch.load(
        SAE_FEATURES_PATH,
        map_location="cpu",
        weights_only=False
    )

    features = sae_data["features"].float()
    labels = sae_data["labels"].numpy()

    analysis = pd.read_csv(
        ANALYSIS_PATH
    )

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    return features, labels, analysis, classifier


def select_safety_features(analysis):

    ranked = analysis.sort_values(
        "difference",
        ascending=False
    )

    selected = ranked.head(
        TOP_FEATURES
    )

    indices = []

    for value in selected["feature"]:

        if isinstance(value, str):
            value = value.replace(
                "feature_",
                ""
            )

        indices.append(
            int(value)
        )

    return indices, selected


def predict_probability(
    classifier,
    features
):

    probabilities = classifier.predict_proba(
        features.numpy()
    )

    return probabilities[:, 1]


def main():

    print("=" * 60)
    print("SAE SAFETY FEATURE INTERVENTION")
    print("=" * 60)

    features, labels, analysis, classifier = load_data()

    print(
        f"Samples: {features.shape[0]}"
    )

    print(
        f"SAE features: {features.shape[1]}"
    )

    safety_indices, selected_features = (
        select_safety_features(
            analysis
        )
    )

    print(
        f"\nSelected safety features: {safety_indices}"
    )

    baseline_probability = predict_probability(
        classifier,
        features
    )

    modified_features = features.clone()

    modified_features[
        :,
        safety_indices
    ] *= (
        1.0 - SUPPRESSION_RATE
    )

    intervention_probability = predict_probability(
        classifier,
        modified_features
    )

    baseline_decision = (
        baseline_probability >= THRESHOLD
    )

    intervention_decision = (
        intervention_probability >= THRESHOLD
    )

    results = pd.DataFrame(
        {
            "label": labels,
            "baseline_safety_probability":
                baseline_probability,
            "intervention_safety_probability":
                intervention_probability,
            "probability_change":
                intervention_probability
                - baseline_probability,
            "baseline_decision":
                np.where(
                    baseline_decision,
                    "BLOCK",
                    "ALLOW"
                ),
            "intervention_decision":
                np.where(
                    intervention_decision,
                    "BLOCK",
                    "ALLOW"
                ),
            "decision_changed":
                baseline_decision
                != intervention_decision
        }
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    harmful_mask = labels == 1
    unharmful_mask = labels == 0

    print(
        "\nBaseline harmful probability:"
    )

    print(
        baseline_probability[harmful_mask].mean()
    )

    print(
        "\nIntervention harmful probability:"
    )

    print(
        intervention_probability[harmful_mask].mean()
    )

    print(
        "\nBaseline unharmful probability:"
    )

    print(
        baseline_probability[unharmful_mask].mean()
    )

    print(
        "\nIntervention unharmful probability:"
    )

    print(
        intervention_probability[unharmful_mask].mean()
    )

    print(
        "\nDecision changes:",
        results["decision_changed"].sum()
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()