import os
import joblib
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "prompt_train_sae_features_full.pt"
)

FP_FEATURES_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "joke_false_positive_features.csv"
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
    "partial_fp_intervention_results.csv"
)

SUPPRESSION_RATES = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
]

CLASSIFIER_THRESHOLD = 0.60

CANDIDATE_COUNT = 10


def load_data():

    data = torch.load(
        FEATURES_PATH,
        map_location="cpu",
        weights_only=False
    )

    features = data["features"].float()
    labels = data["labels"].numpy()

    fp_data = pd.read_csv(
        FP_FEATURES_PATH
    )

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    return features, labels, fp_data, classifier


def get_probability(
    classifier,
    features
):

    return classifier.predict_proba(
        features.numpy()
    )[:, 1]


def get_metrics(
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

    return tp, tn, fp, fn


def main():

    print("=" * 60)
    print("PARTIAL FALSE-POSITIVE INTERVENTION")
    print("=" * 60)

    features, labels, fp_data, classifier = load_data()

    candidates = (
        fp_data
        .groupby("feature")["contribution"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(CANDIDATE_COUNT)
        .index
        .astype(int)
        .tolist()
    )

    print(
        f"Candidate features: {candidates}"
    )

    baseline_probability = get_probability(
        classifier,
        features
    )

    baseline_tp, baseline_tn, baseline_fp, baseline_fn = (
        get_metrics(
            labels,
            baseline_probability
        )
    )

    print(
        f"Baseline TP={baseline_tp} "
        f"TN={baseline_tn} "
        f"FP={baseline_fp} "
        f"FN={baseline_fn}"
    )

    results = []

    for suppression_rate in SUPPRESSION_RATES:

        for feature in candidates:

            modified_features = features.clone()

            modified_features[:, feature] *= (
                1.0 - suppression_rate
            )

            probability = get_probability(
                classifier,
                modified_features
            )

            tp, tn, fp, fn = get_metrics(
                labels,
                probability
            )

            fp_change = (
                fp - baseline_fp
            )

            fn_change = (
                fn - baseline_fn
            )

            results.append(
                {
                    "feature": feature,
                    "suppression_rate":
                        suppression_rate,
                    "baseline_tp":
                        baseline_tp,
                    "baseline_tn":
                        baseline_tn,
                    "baseline_fp":
                        baseline_fp,
                    "baseline_fn":
                        baseline_fn,
                    "intervention_tp":
                        tp,
                    "intervention_tn":
                        tn,
                    "intervention_fp":
                        fp,
                    "intervention_fn":
                        fn,
                    "false_positive_change":
                        fp_change,
                    "false_negative_change":
                        fn_change,
                    "true_positive_change":
                        tp - baseline_tp,
                    "true_negative_change":
                        tn - baseline_tn
                }
            )

            print(
                f"Feature={feature} "
                f"Suppression={suppression_rate:.2f} "
                f"FP={fp} "
                f"FN={fn} "
                f"FP_change={fp_change} "
                f"FN_change={fn_change}"
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

    print("\nBest balanced candidates:")

    balanced = results_df[
        results_df["false_negative_change"] <= 500
    ].sort_values(
        [
            "false_positive_change",
            "false_negative_change"
        ],
        ascending=[
            True,
            True
        ]
    ).head(15)

    print(
        balanced.to_string(
            index=False
        )
    )

    print("=" * 60)


if __name__ == "__main__":
    main()