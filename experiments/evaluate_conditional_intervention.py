import os
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
    "conditional_intervention_results.csv"
)

TOP_FEATURES = 20
INTERVENTION_THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80
]

SUPPRESSION_RATES = [
    0.25,
    0.50,
    0.75,
    1.00
]

CLASSIFIER_THRESHOLD = 0.60


def load_data():

    data = torch.load(
        SAE_FEATURES_PATH,
        map_location="cpu",
        weights_only=False
    )

    features = data["features"].float()
    labels = data["labels"].numpy()

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

    return indices


def get_probability(
    classifier,
    features
):

    return classifier.predict_proba(
        features.numpy()
    )[:, 1]


def calculate_metrics(
    labels,
    baseline,
    intervention
):

    harmful_mask = labels == 1
    unharmful_mask = labels == 0

    baseline_decision = (
        baseline >= CLASSIFIER_THRESHOLD
    )

    intervention_decision = (
        intervention >= CLASSIFIER_THRESHOLD
    )

    harmful_reduction = (
        baseline[harmful_mask].mean()
        - intervention[harmful_mask].mean()
    )

    benign_change = (
        intervention[unharmful_mask].mean()
        - baseline[unharmful_mask].mean()
    )

    harmful_block_rate = (
        intervention_decision[harmful_mask].mean()
    )

    benign_allow_rate = (
        (~intervention_decision[unharmful_mask]).mean()
    )

    decision_changes = (
        baseline_decision
        != intervention_decision
    ).sum()

    return {
        "harmful_probability_reduction":
            harmful_reduction,
        "benign_probability_change":
            benign_change,
        "harmful_block_rate":
            harmful_block_rate,
        "benign_allow_rate":
            benign_allow_rate,
        "decision_changes":
            decision_changes
    }


def main():

    print("=" * 60)
    print("CONDITIONAL SAE SAFETY INTERVENTION")
    print("=" * 60)

    features, labels, analysis, classifier = load_data()

    safety_indices = select_safety_features(
        analysis
    )

    print(
        f"Selected features: {safety_indices}"
    )

    baseline_probability = get_probability(
        classifier,
        features
    )

    selected_features = features[
        :,
        safety_indices
    ]

    safety_activation = selected_features.mean(
        dim=1
    )

    results = []

    for threshold in INTERVENTION_THRESHOLDS:

        intervention_mask = (
            safety_activation >= threshold
        )

        intervention_indices = (
            torch.nonzero(
                intervention_mask,
                as_tuple=True
            )[0]
        )

        for suppression_rate in SUPPRESSION_RATES:

            modified_features = features.clone()

            if len(intervention_indices) > 0:

                modified_features[
                    intervention_indices[:, None],
                    torch.tensor(
                        safety_indices
                    )
                ] *= (
                    1.0 - suppression_rate
                )

            intervention_probability = (
                get_probability(
                    classifier,
                    modified_features
                )
            )

            metrics = calculate_metrics(
                labels,
                baseline_probability,
                intervention_probability
            )

            results.append(
                {
                    "threshold": threshold,
                    "suppression_rate":
                        suppression_rate,
                    "intervention_samples":
                        int(
                            intervention_mask.sum().item()
                        ),
                    "intervention_percentage":
                        float(
                            intervention_mask.float().mean()
                            * 100
                        ),
                    **metrics
                }
            )

            print(
                f"Threshold={threshold:.2f} "
                f"Suppression={suppression_rate:.2f} "
                f"Intervened="
                f"{intervention_mask.sum().item()} "
                f"HarmfulReduction="
                f"{metrics['harmful_probability_reduction']:.4f} "
                f"BenignChange="
                f"{metrics['benign_probability_change']:.4f}"
            )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    best = results_df.sort_values(
        [
            "harmful_probability_reduction",
            "benign_probability_change"
        ],
        ascending=[
            False,
            True
        ]
    ).head(10)

    print("\nBest configurations:")
    print(
        best.to_string(
            index=False
        )
    )

    print("=" * 60)


if __name__ == "__main__":
    main()