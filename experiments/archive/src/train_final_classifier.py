import torch
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


WILDGUARD_PATH = "data/processed/prompt_train_sae_features_full.pt"
OASST_PATH = "data/processed/oasst_benign_sae_features.pt"
HARD_NEGATIVE_PATH = "data/processed/oasst_hard_negative_sae_features.pt"
CONVERSATION_PATH = "data/processed/oasst_conversation_sae_features.pt"
TARGETED_PATH = "data/processed/targeted_cyber_sae_features.pt"

OUTPUT_PATH = "models/final_conversation_weighted_classifier.pkl"

CONVERSATION_WEIGHT = 10.0


def load_features(path):

    data = torch.load(
        path,
        map_location="cpu"
    )

    features = data["features"]
    labels = data["labels"]

    if not torch.is_tensor(labels):

        labels = torch.tensor(
            [
                1 if str(x).lower() == "harmful" else 0
                for x in labels
            ]
        )

    labels = labels.long()

    return (
        features.numpy(),
        labels.numpy()
    )


X_wildguard, y_wildguard = load_features(
    WILDGUARD_PATH
)

X_oasst, y_oasst = load_features(
    OASST_PATH
)

X_hard, y_hard = load_features(
    HARD_NEGATIVE_PATH
)

X_conversation, y_conversation = load_features(
    CONVERSATION_PATH
)

X_targeted, y_targeted = load_features(
    TARGETED_PATH
)


X = np.concatenate(
    [
        X_wildguard,
        X_oasst,
        X_hard,
        X_conversation,
        X_targeted
    ],
    axis=0
)

y = np.concatenate(
    [
        y_wildguard,
        y_oasst,
        y_hard,
        y_conversation,
        y_targeted
    ],
    axis=0
)

sample_weights = np.ones(
    len(y),
    dtype=np.float64
)

conversation_start = len(y_wildguard) + len(y_oasst) + len(y_hard)

conversation_end = (
    conversation_start + len(y_conversation)
)

sample_weights[
    conversation_start:conversation_end
] = CONVERSATION_WEIGHT


print(
    f"WildGuardMix features: {X_wildguard.shape}"
)

print(
    f"OASST1 features: {X_oasst.shape}"
)

print(
    f"Hard-negative features: {X_hard.shape}"
)

print(
    f"Conversation features: {X_conversation.shape}"
)

print(
    f"Targeted features: {X_targeted.shape}"
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

print(
    f"Conversation sample weight: {CONVERSATION_WEIGHT}"
)


classifier = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


classifier.fit(
    X,
    y,
    classifier__sample_weight=sample_weights
)


joblib.dump(
    classifier,
    OUTPUT_PATH
)


print()
print(
    "Conversation-weighted classifier training completed."
)

print(
    f"Training samples: {len(y)}"
)

print(
    f"Features: {X.shape[1]}"
)

print(
    f"Classifier saved to: {OUTPUT_PATH}"
)