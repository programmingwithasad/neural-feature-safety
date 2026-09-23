import torch
import numpy as np
import pandas as pd
import joblib


WILDGUARD_PATH = "data/processed/prompt_train_sae_features_full.pt"
OASST_PATH = "data/processed/oasst_benign_sae_features.pt"
HARD_NEGATIVE_PATH = "data/processed/oasst_hard_negative_sae_features.pt"
CLASSIFIER_PATH = "models/conversation_enhanced_safety_classifier.pkl"
OUTPUT_PATH = "data/processed/feature_specificity_analysis.csv"


def load_features(path):
    data = torch.load(
        path,
        map_location="cpu"
    )

    features = data["features"]

    if torch.is_tensor(features):
        features = features.numpy()

    labels = data.get("labels")

    if labels is None:
        labels = np.zeros(
            len(features),
            dtype=np.int64
        )
    elif torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    labels = np.asarray(labels)

    if not np.issubdtype(labels.dtype, np.number):
        labels = np.array(
            [
                1 if str(x).lower() == "harmful" else 0
                for x in labels
            ],
            dtype=np.int64
        )
    else:
        labels = labels.astype(np.int64)

    return features, labels


X_wildguard, y_wildguard = load_features(
    WILDGUARD_PATH
)

X_oasst, y_oasst = load_features(
    OASST_PATH
)

X_hard, y_hard = load_features(
    HARD_NEGATIVE_PATH
)


X = np.concatenate(
    [
        X_wildguard,
        X_oasst,
        X_hard
    ],
    axis=0
)

y = np.concatenate(
    [
        y_wildguard,
        y_oasst,
        y_hard
    ],
    axis=0
)


classifier = joblib.load(
    CLASSIFIER_PATH
)

scaler = classifier.named_steps["scaler"]
logistic = classifier.named_steps["classifier"]

weights = logistic.coef_[0]


harmful_mask = y == 1
unharmful_mask = y == 0


harmful_features = X[harmful_mask]
unharmful_features = X[unharmful_mask]


harmful_mean = harmful_features.mean(
    axis=0
)

unharmful_mean = unharmful_features.mean(
    axis=0
)

harmful_median = np.median(
    harmful_features,
    axis=0
)

unharmful_median = np.median(
    unharmful_features,
    axis=0
)

harmful_activation_rate = (
    (harmful_features > 0).mean(axis=0)
)

unharmful_activation_rate = (
    (unharmful_features > 0).mean(axis=0)
)

mean_difference = (
    harmful_mean - unharmful_mean
)

activation_rate_difference = (
    harmful_activation_rate
    - unharmful_activation_rate
)

absolute_difference = np.abs(
    mean_difference
)

specificity_score = (
    mean_difference
    * activation_rate_difference
)

scaled_weights = (
    weights / scaler.scale_
)


results = pd.DataFrame(
    {
        "feature": np.arange(
            X.shape[1]
        ),
        "harmful_mean": harmful_mean,
        "unharmful_mean": unharmful_mean,
        "harmful_median": harmful_median,
        "unharmful_median": unharmful_median,
        "harmful_activation_rate": harmful_activation_rate,
        "unharmful_activation_rate": unharmful_activation_rate,
        "mean_difference": mean_difference,
        "activation_rate_difference": activation_rate_difference,
        "absolute_difference": absolute_difference,
        "specificity_score": specificity_score,
        "classifier_weight": weights,
        "scaled_classifier_weight": scaled_weights
    }
)


results["safety_alignment"] = (
    results["mean_difference"]
    * results["classifier_weight"]
)


results["specificity_alignment"] = (
    results["specificity_score"]
    * results["classifier_weight"]
)


results = results.sort_values(
    "specificity_alignment",
    ascending=False
)


results.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"Total features analyzed: {len(results)}"
)

print()
print("Top 20 safety-specific features")
print(
    results[
        [
            "feature",
            "harmful_mean",
            "unharmful_mean",
            "harmful_activation_rate",
            "unharmful_activation_rate",
            "specificity_score",
            "classifier_weight"
        ]
    ].head(20).to_string(
        index=False
    )
)

print()
print("Top 20 potential false-positive features")

false_positive = results[
    (results["classifier_weight"] > 0)
    & (
        results["unharmful_mean"]
        > results["harmful_mean"]
    )
].copy()

print(
    false_positive[
        [
            "feature",
            "harmful_mean",
            "unharmful_mean",
            "harmful_activation_rate",
            "unharmful_activation_rate",
            "specificity_score",
            "classifier_weight"
        ]
    ].head(20).to_string(
        index=False
    )
)

print()
print("Top 20 strongly aligned safety features")

aligned = results[
    (results["classifier_weight"] > 0)
    & (
        results["mean_difference"] > 0
    )
].copy()

print(
    aligned[
        [
            "feature",
            "harmful_mean",
            "unharmful_mean",
            "harmful_activation_rate",
            "unharmful_activation_rate",
            "mean_difference",
            "classifier_weight",
            "safety_alignment"
        ]
    ].head(20).to_string(
        index=False
    )
)

print()
print(
    f"Saved to: {OUTPUT_PATH}"
)