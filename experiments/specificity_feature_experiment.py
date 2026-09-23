import torch
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


TRAIN_PATH = "data/processed/prompt_train_sae_features_full.pt"
VALIDATION_PATH = "data/processed/prompt_validation_sae_features.pt"
TEST_PATH = "data/processed/prompt_test_sae_features.pt"
SPECIFICITY_PATH = "data/processed/feature_specificity_analysis.csv"


def load_data(path):
    data = torch.load(
        path,
        map_location="cpu"
    )

    X = data["features"]

    if torch.is_tensor(X):
        X = X.numpy()

    y = data["labels"]

    if torch.is_tensor(y):
        y = y.numpy()

    y = np.asarray(y).astype(np.int64)

    return X, y


X_train, y_train = load_data(
    TRAIN_PATH
)

X_validation, y_validation = load_data(
    VALIDATION_PATH
)

X_test, y_test = load_data(
    TEST_PATH
)


specificity = pd.read_csv(
    SPECIFICITY_PATH
)

specificity = specificity.sort_values(
    "specificity_alignment",
    ascending=False
)

top_k_values = [
    25,
    50,
    100,
    250,
    500,
    1000,
    1500,
    2048
]


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Validation samples: {len(X_validation)}"
)

print(
    f"Test samples: {len(X_test)}"
)

print()
print(
    "SPECIFICITY-BASED FEATURE EXPERIMENT"
)

print(
    "=" * 70
)


results = []


for k in top_k_values:

    selected_features = (
        specificity[
            "feature"
        ].head(k).to_numpy()
    )

    selected_features = np.sort(
        selected_features
    )

    Xtr = X_train[
        :,
        selected_features
    ]

    Xv = X_validation[
        :,
        selected_features
    ]

    Xt = X_test[
        :,
        selected_features
    ]

    classifier = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    classifier.fit(
        Xtr,
        y_train
    )

    validation_probabilities = (
        classifier.predict_proba(
            Xv
        )[:, 1]
    )

    validation_predictions = (
        validation_probabilities >= 0.5
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

    test_probabilities = (
        classifier.predict_proba(
            Xt
        )[:, 1]
    )

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(np.int64)

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

    results.append(
        {
            "top_k": k,
            "validation_accuracy": validation_accuracy,
            "validation_precision": validation_precision,
            "validation_recall": validation_recall,
            "validation_f1": validation_f1,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }
    )

    print()
    print(
        f"Top {k}"
    )

    print(
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )

    print(
        f"Validation Precision: {validation_precision:.4f}"
    )

    print(
        f"Validation Recall: {validation_recall:.4f}"
    )

    print(
        f"Validation F1: {validation_f1:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )

    print(
        f"Test Precision: {test_precision:.4f}"
    )

    print(
        f"Test Recall: {test_recall:.4f}"
    )

    print(
        f"Test F1: {test_f1:.4f}"
    )


results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "data/processed/specificity_feature_experiment.csv",
    index=False
)


best = results_df.loc[
    results_df[
        "validation_f1"
    ].idxmax()
]


print()
print(
    "=" * 70
)

print(
    "BEST VALIDATION MODEL"
)

print(
    f"Top-K: {int(best['top_k'])}"
)

print(
    f"Validation F1: {best['validation_f1']:.4f}"
)

print(
    f"Test Accuracy: {best['test_accuracy']:.4f}"
)

print(
    f"Test Precision: {best['test_precision']:.4f}"
)

print(
    f"Test Recall: {best['test_recall']:.4f}"
)

print(
    f"Test F1: {best['test_f1']:.4f}"
)

print()
print(
    "Saved to: data/processed/specificity_feature_experiment.csv"
)