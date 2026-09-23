import torch
import numpy as np
import pandas as pd
import joblib


FEATURE_SETS = [
    [518, 554],
    [518, 89],
    [554, 89],
    [518, 554, 89],
    [518, 554, 1380],
    [518, 89, 1380],
    [554, 89, 1380],
    [518, 554, 89, 1380]
]

SUPPRESSION_RATES = [
    0.02,
    0.03,
    0.05,
    0.07,
    0.10
]

THRESHOLD = 0.50

data = torch.load(
    "data/processed/prompt_validation_sae_features.pt",
    map_location="cpu",
    weights_only=False
)

X = data["features"].numpy()
y = data["labels"].numpy()

classifier = joblib.load(
    "models/final_safety_classifier.pkl"
)

baseline_probability = classifier.predict_proba(X)[:, 1]
baseline_prediction = baseline_probability >= THRESHOLD

baseline_tp = int(np.sum((y == 1) & baseline_prediction))
baseline_tn = int(np.sum((y == 0) & ~baseline_prediction))
baseline_fp = int(np.sum((y == 0) & baseline_prediction))
baseline_fn = int(np.sum((y == 1) & ~baseline_prediction))

print("=" * 60)
print("COMBINED SAE FP INTERVENTION")
print("=" * 60)

print(
    f"Baseline TP={baseline_tp} "
    f"TN={baseline_tn} "
    f"FP={baseline_fp} "
    f"FN={baseline_fn}"
)

results = []

for feature_set in FEATURE_SETS:

    for suppression in SUPPRESSION_RATES:

        X_modified = X.copy()

        for feature in feature_set:

            X_modified[:, feature] *= (
                1.0 - suppression
            )

        probability = classifier.predict_proba(
            X_modified
        )[:, 1]

        prediction = probability >= THRESHOLD

        tp = int(np.sum((y == 1) & prediction))
        tn = int(np.sum((y == 0) & ~prediction))
        fp = int(np.sum((y == 0) & prediction))
        fn = int(np.sum((y == 1) & ~prediction))

        harmful_became_allowed = int(
            np.sum(
                (y == 1)
                & baseline_prediction
                & ~prediction
            )
        )

        benign_became_blocked = int(
            np.sum(
                (y == 0)
                & ~baseline_prediction
                & prediction
            )
        )

        fp_change = fp - baseline_fp
        fn_change = fn - baseline_fn

        accuracy = (
            (tp + tn) / len(y)
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        decision_changes = int(
            np.sum(
                baseline_prediction != prediction
            )
        )

        print(
            f"Features={feature_set} "
            f"Suppression={suppression:.2f} "
            f"FP={fp} "
            f"FN={fn} "
            f"FP_change={fp_change} "
            f"FN_change={fn_change} "
            f"HarmfulBlockToAllow={harmful_became_allowed}"
        )

        results.append(
            {
                "features": str(feature_set),
                "suppression_rate": suppression,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "fp_change": fp_change,
                "fn_change": fn_change,
                "harmful_became_allowed": harmful_became_allowed,
                "benign_became_blocked": benign_became_blocked,
                "decision_changes": decision_changes
            }
        )

results_df = pd.DataFrame(results)

safe_results = results_df[
    results_df["harmful_became_allowed"] == 0
].copy()

safe_results = safe_results.sort_values(
    ["fp_change", "f1"],
    ascending=[True, False]
)

print()
print("=" * 60)
print("SAFE COMBINED CONFIGURATIONS")
print("=" * 60)

if len(safe_results) == 0:

    print("No completely safe configurations found.")

else:

    print(
        safe_results[
            [
                "features",
                "suppression_rate",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "fp",
                "fn",
                "fp_change",
                "fn_change",
                "harmful_became_allowed",
                "benign_became_blocked",
                "decision_changes"
            ]
        ].head(30).to_string(index=False)
    )

output_path = (
    "data/processed/"
    "combined_fp_intervention_validation.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print()
print("Results saved to:")
print(output_path)
print("=" * 60)