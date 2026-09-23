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
    "fp_candidate_intervention_results.csv"
)

CANDIDATES = [
    1049,
    1151,
    146,
    518,
    1394,
    89,
    318,
    791,
    554,
    1380
]

SUPPRESSION_RATES = [
    0.05,
    0.10,
    0.15,
    0.20
]

THRESHOLD = 0.60


def calculate_metrics(labels, probabilities):

    predictions = probabilities >= THRESHOLD

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
    print("FP CANDIDATE FEATURE INTERVENTION")
    print("=" * 60)

    data = torch.load(
        FEATURES_PATH,
        map_location="cpu",
        weights_only=False
    )

    features = data["features"].float()
    labels = data["labels"].numpy()

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    baseline_probability = classifier.predict_proba(
        features.numpy()
    )[:, 1]

    baseline_tp, baseline_tn, baseline_fp, baseline_fn = (
        calculate_metrics(
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

    for feature in CANDIDATES:

        for suppression_rate in SUPPRESSION_RATES:

            modified_features = features.clone()

            modified_features[:, feature] *= (
                1.0 - suppression_rate
            )

            probability = classifier.predict_proba(
                modified_features.numpy()
            )[:, 1]

            tp, tn, fp, fn = calculate_metrics(
                labels,
                probability
            )

            fp_reduction = (
                baseline_fp - fp
            )

            fn_increase = (
                fn - baseline_fn
            )

            tp_loss = (
                baseline_tp - tp
            )

            score = (
                fp_reduction
                - 2.0 * fn_increase
            )

            results.append(
                {
                    "feature": feature,
                    "suppression_rate":
                        suppression_rate,
                    "baseline_fp":
                        baseline_fp,
                    "baseline_fn":
                        baseline_fn,
                    "intervention_fp":
                        fp,
                    "intervention_fn":
                        fn,
                    "baseline_tp":
                        baseline_tp,
                    "intervention_tp":
                        tp,
                    "baseline_tn":
                        baseline_tn,
                    "intervention_tn":
                        tn,
                    "fp_reduction":
                        fp_reduction,
                    "fn_increase":
                        fn_increase,
                    "tp_loss":
                        tp_loss,
                    "tn_gain":
                        tn - baseline_tn,
                    "safety_tradeoff":
                        fp_reduction / max(
                            tp_loss,
                            1
                        ),
                    "score":
                        score
                }
            )

            print(
                f"Feature={feature} "
                f"Suppression={suppression_rate:.2f} "
                f"FP={fp} "
                f"FN={fn} "
                f"TP={tp} "
                f"FP_reduction={fp_reduction} "
                f"FN_increase={fn_increase}"
            )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        [
            "score",
            "fp_reduction"
        ],
        ascending=[
            False,
            False
        ]
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nBest candidates:")
    print(
        results_df.head(15).to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()