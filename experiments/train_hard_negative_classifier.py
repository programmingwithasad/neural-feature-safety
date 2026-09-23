import torch
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


WILDGUARD_PATH = "data/processed/prompt_train_sae_features_full.pt"
OASST_PATH = "data/processed/oasst_benign_sae_features.pt"
HARD_NEGATIVE_PATH = "data/processed/oasst_hard_negative_sae_features.pt"
OUTPUT_PATH = "models/hard_negative_safety_classifier.pkl"


wildguard = torch.load(
    WILDGUARD_PATH,
    map_location="cpu"
)

wildguard_features = wildguard["features"].numpy()
wildguard_labels = wildguard["labels"].numpy().astype(np.int64)


oasst = torch.load(
    OASST_PATH,
    map_location="cpu"
)

oasst_features = oasst["features"].numpy()
oasst_labels = np.zeros(
    len(oasst_features),
    dtype=np.int64
)


hard_negative = torch.load(
    HARD_NEGATIVE_PATH,
    map_location="cpu"
)

hard_negative_features = hard_negative["features"].numpy()
hard_negative_labels = np.zeros(
    len(hard_negative_features),
    dtype=np.int64
)


X = np.concatenate(
    [
        wildguard_features,
        oasst_features,
        hard_negative_features
    ],
    axis=0
)

y = np.concatenate(
    [
        wildguard_labels,
        oasst_labels,
        hard_negative_labels
    ],
    axis=0
)


print(
    f"WildGuardMix features: {wildguard_features.shape}"
)

print(
    f"OASST1 features: {oasst_features.shape}"
)

print(
    f"Hard-negative features: {hard_negative_features.shape}"
)

print(
    f"Combined training features: {X.shape}"
)

print(
    f"Harmful samples: {(y == 1).sum()}"
)

print(
    f"Unharmful samples: {(y == 0).sum()}"
)


classifier = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
)

classifier.fit(X, y)


joblib.dump(
    classifier,
    OUTPUT_PATH
)


print()
print("Hard-negative classifier training completed.")
print(f"Training samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Classifier saved to: {OUTPUT_PATH}")