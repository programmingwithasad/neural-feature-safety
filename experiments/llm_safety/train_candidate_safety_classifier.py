import os
import torch
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

paths = {
    "wildguard": os.path.join(
        project_root,
        "data",
        "processed",
        "prompt_train_sae_features_full.pt"
    ),
    "oasst": os.path.join(
        project_root,
        "data",
        "processed",
        "oasst_benign_sae_features.pt"
    ),
    "hard_negative": os.path.join(
        project_root,
        "data",
        "processed",
        "oasst_hard_negative_sae_features.pt"
    ),
    "conversation": os.path.join(
        project_root,
        "data",
        "processed",
        "oasst_conversation_sae_features.pt"
    ),
    "targeted": os.path.join(
        project_root,
        "data",
        "processed",
        "targeted_cyber_sae_features.pt"
    ),
    "augmentation": os.path.join(
        project_root,
        "data",
        "processed",
        "refusal_training_augmentation_features.pt"
    )
}

output_path = os.path.join(
    project_root,
    "models",
    "candidate_safety_classifier_refusal_v1.pkl"
)


def load_dataset(path, force_label=None):

    data = torch.load(
        path,
        map_location="cpu"
    )

    features = data["features"]

    if torch.is_tensor(features):
        features = features.numpy()
    else:
        features = np.asarray(features)

    labels = data["labels"]

    if force_label is not None:
        labels = np.full(
            len(features),
            force_label,
            dtype=np.int64
        )

    elif torch.is_tensor(labels):
        labels = labels.numpy()

    else:
        labels = np.asarray(labels)

        if labels.dtype.kind not in "iu":
            labels = np.array(
                [
                    1 if str(x).lower() == "harmful" else 0
                    for x in labels
                ],
                dtype=np.int64
            )

    labels = labels.astype(np.int64)

    return features, labels


X_wildguard, y_wildguard = load_dataset(
    paths["wildguard"]
)

X_oasst, y_oasst = load_dataset(
    paths["oasst"],
    force_label=0
)

X_hard, y_hard = load_dataset(
    paths["hard_negative"],
    force_label=0
)

X_conversation, y_conversation = load_dataset(
    paths["conversation"],
    force_label=0
)

X_targeted, y_targeted = load_dataset(
    paths["targeted"]
)

X_aug, y_aug = load_dataset(
    paths["augmentation"]
)


X = np.concatenate(
    [
        X_wildguard,
        X_oasst,
        X_hard,
        X_conversation,
        X_targeted,
        X_aug
    ],
    axis=0
)

y = np.concatenate(
    [
        y_wildguard,
        y_oasst,
        y_hard,
        y_conversation,
        y_targeted,
        y_aug
    ],
    axis=0
)


sample_weights = np.ones(
    len(y),
    dtype=np.float64
)


augmentation_start = (
    len(y_wildguard)
    + len(y_oasst)
    + len(y_hard)
    + len(y_conversation)
    + len(y_targeted)
)


augmentation_end = (
    augmentation_start
    + len(y_aug)
)


sample_weights[
    augmentation_start:augmentation_end
] = 10.0


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
                C=1.0,
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


os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)

joblib.dump(
    classifier,
    output_path
)


print("=" * 70)
print("CANDIDATE SAFETY CLASSIFIER")
print("=" * 70)

print(
    f"WildGuard features: "
    f"{X_wildguard.shape}"
)

print(
    f"OASST benign features: "
    f"{X_oasst.shape}"
)

print(
    f"Hard-negative features: "
    f"{X_hard.shape}"
)

print(
    f"Conversation features: "
    f"{X_conversation.shape}"
)

print(
    f"Targeted features: "
    f"{X_targeted.shape}"
)

print(
    f"Refusal augmentation: "
    f"{X_aug.shape}"
)

print(
    f"\nCombined features: "
    f"{X.shape}"
)

print(
    f"Harmful samples: "
    f"{int((y == 1).sum())}"
)

print(
    f"Unharmful samples: "
    f"{int((y == 0).sum())}"
)

print(
    f"Augmentation sample weight: "
    f"{sample_weights[augmentation_start]}"
)

print(
    f"\nCandidate saved to:\n"
    f"{output_path}"
)