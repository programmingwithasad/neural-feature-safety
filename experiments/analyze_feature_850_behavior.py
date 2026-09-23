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
    "feature_850_behavior_analysis.csv"
)

FEATURE_INDEX = 850
CLASSIFIER_THRESHOLD = 0.60


def load_data():

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

    return features, labels, classifier


def main():

    print("=" * 60)
    print("FEATURE 850 BEHAVIOR ANALYSIS")
    print("=" * 60)

    features, labels, classifier = load_data()

    probabilities = classifier.predict_proba(
        features
    )[:, 1]

    predictions = (
        probabilities >= CLASSIFIER_THRESHOLD
    )

    feature_values = features[
        :,
        FEATURE_INDEX
    ]

    categories = np.full(
        len(labels),
        "UNKNOWN",
        dtype=object
    )

    categories[
        (labels == 1) & predictions
    ] = "TRUE_POSITIVE"

    categories[
        (labels == 0) & ~predictions
    ] = "TRUE_NEGATIVE"

    categories[
        (labels == 0) & predictions
    ] = "FALSE_POSITIVE"

    categories[
        (labels == 1) & ~predictions
    ] = "FALSE_NEGATIVE"

    results = []

    for category in [
        "TRUE_POSITIVE",
        "TRUE_NEGATIVE",
        "FALSE_POSITIVE",
        "FALSE_NEGATIVE"
    ]:

        mask = categories == category

        values = feature_values[
            mask
        ]

        results.append(
            {
                "category": category,
                "samples": len(values),
                "mean_activation":
                    values.mean(),
                "median_activation":
                    np.median(values),
                "std_activation":
                    values.std(),
                "minimum_activation":
                    values.min(),
                "maximum_activation":
                    values.max(),
                "activation_above_0.5":
                    np.mean(
                        values >= 0.5
                    ),
                "activation_above_0.75":
                    np.mean(
                        values >= 0.75
                    ),
                "activation_above_1.0":
                    np.mean(
                        values >= 1.0
                    )
            }
        )

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    fp_mean = results_df.loc[
        results_df["category"]
        == "FALSE_POSITIVE",
        "mean_activation"
    ].iloc[0]

    tp_mean = results_df.loc[
        results_df["category"]
        == "TRUE_POSITIVE",
        "mean_activation"
    ].iloc[0]

    print(
        "\nFalse-positive vs true-positive mean activation ratio:"
    )

    print(
        fp_mean / tp_mean
        if tp_mean != 0
        else 0
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()