import os
import joblib
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FEATURES_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "prompt_test_sae_features.pt"
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
    "all_sae_feature_fp_analysis.csv"
)

THRESHOLD = 0.60


def main():

    print("=" * 60)
    print("ALL SAE FEATURE FP ANALYSIS")
    print("=" * 60)

    data = torch.load(
        FEATURES_PATH,
        map_location="cpu",
        weights_only=False
    )

    features = data["features"].float().numpy()
    labels = data["labels"].numpy()

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    probabilities = classifier.predict_proba(
        features
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    )

    tp_mask = (
        (labels == 1)
        & predictions
    )

    tn_mask = (
        (labels == 0)
        & ~predictions
    )

    fp_mask = (
        (labels == 0)
        & predictions
    )

    fn_mask = (
        (labels == 1)
        & ~predictions
    )

    print(
        f"Features: {features.shape[1]}"
    )

    print(
        f"TP samples: {tp_mask.sum()}"
    )

    print(
        f"TN samples: {tn_mask.sum()}"
    )

    print(
        f"FP samples: {fp_mask.sum()}"
    )

    print(
        f"FN samples: {fn_mask.sum()}"
    )

    rows = []

    for feature in range(
        features.shape[1]
    ):

        fp_values = features[
            fp_mask,
            feature
        ]

        tp_values = features[
            tp_mask,
            feature
        ]

        tn_values = features[
            tn_mask,
            feature
        ]

        fn_values = features[
            fn_mask,
            feature
        ]

        fp_mean = fp_values.mean()
        tp_mean = tp_values.mean()

        fp_tp_difference = (
            fp_mean - tp_mean
        )

        fp_tp_ratio = (
            fp_mean / tp_mean
            if tp_mean != 0
            else 0
        )

        fp_high_activation = np.mean(
            fp_values >= 0.5
        )

        tp_high_activation = np.mean(
            tp_values >= 0.5
        )

        rows.append(
            {
                "feature": feature,
                "fp_mean": fp_mean,
                "tp_mean": tp_mean,
                "tn_mean": tn_values.mean(),
                "fn_mean": fn_values.mean(),
                "fp_tp_difference":
                    fp_tp_difference,
                "fp_tp_ratio":
                    fp_tp_ratio,
                "fp_activation_above_0.5":
                    fp_high_activation,
                "tp_activation_above_0.5":
                    tp_high_activation,
                "activation_separation":
                    abs(fp_tp_difference)
            }
        )

    results = pd.DataFrame(
        rows
    )

    results = results.sort_values(
        "fp_tp_difference",
        ascending=False
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nTop features with higher FP activation than TP:"
    )

    print(
        results.head(25).to_string(
            index=False
        )
    )

    print(
        "\nTop features with lower FP activation than TP:"
    )

    print(
        results.sort_values(
            "fp_tp_difference"
        ).head(25).to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print("=" * 60)


if __name__ == "__main__":
    main()