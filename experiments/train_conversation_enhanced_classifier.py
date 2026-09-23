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
MODEL_PATH = "models/conversation_enhanced_safety_classifier.pkl"


def normalize_labels(labels):
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    labels = np.asarray(labels)

    if np.issubdtype(labels.dtype, np.number):
        return labels.astype(np.int64)

    normalized = np.array(
        [
            1 if str(label).lower() == "harmful" else 0
            for label in labels
        ],
        dtype=np.int64
    )

    return normalized


wildguard = torch.load(
    WILDGUARD_PATH,
    map_location="cpu"
)

oasst = torch.load(
    OASST_PATH,
    map_location="cpu"
)

hard_negative = torch.load(
    HARD_NEGATIVE_PATH,
    map_location="cpu"
)

conversation = torch.load(
    CONVERSATION_PATH,
    map_location="cpu"
)


X_wildguard = wildguard["features"].numpy()
y_wildguard = normalize_labels(
    wildguard["labels"]
)

X_oasst = oasst["features"].numpy()
y_oasst = normalize_labels(
    oasst["labels"]
)

X_hard_negative = hard_negative["features"].numpy()
y_hard_negative = normalize_labels(
    hard_negative["labels"]
)

X_conversation = conversation["features"].numpy()
y_conversation = normalize_labels(
    conversation["labels"]
)


X = np.concatenate(
    [
        X_wildguard,
        X_oasst,
        X_hard_negative,
        X_conversation
    ],
    axis=0
)

y = np.concatenate(
    [
        y_wildguard,
        y_oasst,
        y_hard_negative,
        y_conversation
    ],
    axis=0
)


print(f"WildGuardMix features: {X_wildguard.shape}")
print(f"OASST1 features: {X_oasst.shape}")
print(f"Hard-negative features: {X_hard_negative.shape}")
print(f"Conversation features: {X_conversation.shape}")
print(f"Combined training features: {X.shape}")
print(f"Harmful samples: {np.sum(y == 1)}")
print(f"Unharmful samples: {np.sum(y == 0)}")


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
    y
)

joblib.dump(
    classifier,
    MODEL_PATH
)

print()
print("Conversation-enhanced classifier training completed.")
print(f"Training samples: {len(y)}")
print(f"Features: {X.shape[1]}")
print(f"Classifier saved to: {MODEL_PATH}")