import torch
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression


WILDGUARD_PATH = "data/processed/prompt_train_sae_features_full.pt"
OASST_PATH = "data/processed/oasst_benign_sae_features.pt"
OUTPUT_PATH = "models/combined_safety_classifier.pkl"


wildguard = torch.load(
    WILDGUARD_PATH,
    map_location="cpu"
)

oasst = torch.load(
    OASST_PATH,
    map_location="cpu"
)

wildguard_features = wildguard["features"].numpy()
wildguard_labels = wildguard["labels"].numpy()

oasst_features = oasst["features"].numpy()
oasst_labels = oasst["labels"].numpy()

X_train = np.concatenate(
    [
        wildguard_features,
        oasst_features
    ],
    axis=0
)

y_train = np.concatenate(
    [
        wildguard_labels,
        oasst_labels
    ],
    axis=0
)

print(
    "WildGuardMix features:",
    wildguard_features.shape
)

print(
    "OASST1 features:",
    oasst_features.shape
)

print(
    "Combined training features:",
    X_train.shape
)

print(
    "Harmful samples:",
    np.sum(y_train == 1)
)

print(
    "Unharmful samples:",
    np.sum(y_train == 0)
)

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
)

classifier.fit(
    X_train,
    y_train
)

with open(
    OUTPUT_PATH,
    "wb"
) as file:

    pickle.dump(
        classifier,
        file
    )

print(
    "\nCombined classifier training completed."
)

print(
    "Training samples:",
    len(y_train)
)

print(
    "Features:",
    X_train.shape[1]
)

print(
    "Classifier saved to:",
    OUTPUT_PATH
)