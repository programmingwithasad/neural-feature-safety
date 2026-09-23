import torch
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


VALIDATION_PATH = "data/processed/prompt_validation_sae_features.pt"
TEST_PATH = "data/processed/prompt_test_sae_features.pt"

MODEL_PATH = "models/final_conversation_weighted_classifier.pkl"

OUTPUT_PATH = "data/processed/final_threshold_sweep.csv"


def load_data(path):

    data = torch.load(
        path,
        map_location="cpu"
    )

    X = data["features"]
    y = data["labels"]

    if torch.is_tensor(X):
        X = X.numpy()

    if torch.is_tensor(y):
        y = y.numpy()

    y = np.asarray(y).astype(np.int64)

    return X, y


X_validation, y_validation = load_data(
    VALIDATION_PATH
)

X_test, y_test = load_data(
    TEST_PATH
)


classifier = joblib.load(
    MODEL_PATH
)


validation_probabilities = classifier.predict_proba(
    X_validation
)[:, 1]

test_probabilities = classifier.predict_proba(
    X_test
)[:, 1]


thresholds = np.arange(
    0.10,
    0.91,
    0.05
)


results = []


print(
    "FINAL CLASSIFIER THRESHOLD SWEEP"
)

print(
    "=" * 80
)

print(
    f"Validation samples: {len(y_validation)}"
)

print(
    f"Test samples: {len(y_test)}"
)

print()


for threshold in thresholds:

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(np.int64)

    test_predictions = (
        test_probabilities >= threshold
    ).astype(np.int64)


    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    validation_precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    validation_recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    validation_f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )


    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )


    matrix = confusion_matrix(
        y_test,
        test_predictions
    )

    tn, fp, fn, tp = matrix.ravel()


    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )


    results.append(
        {
            "threshold": threshold,
            "validation_accuracy": validation_accuracy,
            "validation_precision": validation_precision,
            "validation_recall": validation_recall,
            "validation_f1": validation_f1,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        }
    )


    print(
        f"Threshold {threshold:.2f} | "
        f"Val F1 {validation_f1:.4f} | "
        f"Test F1 {test_f1:.4f} | "
        f"Test Recall {test_recall:.4f} | "
        f"FPR {false_positive_rate:.4f}"
    )


results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


best = results_df.loc[
    results_df[
        "validation_f1"
    ].idxmax()
]


print()
print(
    "=" * 80
)

print(
    "BEST THRESHOLD BY VALIDATION F1"
)

print(
    f"Threshold: {best['threshold']:.2f}"
)

print(
    f"Validation Accuracy: "
    f"{best['validation_accuracy']:.4f}"
)

print(
    f"Validation Precision: "
    f"{best['validation_precision']:.4f}"
)

print(
    f"Validation Recall: "
    f"{best['validation_recall']:.4f}"
)

print(
    f"Validation F1: "
    f"{best['validation_f1']:.4f}"
)

print()

print(
    "Corresponding test performance:"
)

print(
    f"Test Accuracy: "
    f"{best['test_accuracy']:.4f}"
)

print(
    f"Test Precision: "
    f"{best['test_precision']:.4f}"
)

print(
    f"Test Recall: "
    f"{best['test_recall']:.4f}"
)

print(
    f"Test F1: "
    f"{best['test_f1']:.4f}"
)

print(
    f"False Positive Rate: "
    f"{best['false_positive_rate']:.4f}"
)

print(
    f"False Negative Rate: "
    f"{best['false_negative_rate']:.4f}"
)

print()

print(
    f"Saved to: {OUTPUT_PATH}"
)