import torch
import pickle
import numpy as np

from transformers import AutoTokenizer, AutoModel

from autoencoder import SparseAutoencoder


DATA_PATH = "data/processed/oasst_benign_validation.csv"
SAE_PATH = "models/sparse_autoencoder_full.pt"
CLASSIFIER_PATH = "models/combined_safety_classifier.pkl"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = __import__("pandas").read_csv(
    DATA_PATH
)

texts = data["text"].fillna("").tolist()

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

encoder = AutoModel.from_pretrained(
    "distilbert-base-uncased"
).to(device)

encoder.eval()

sae = SparseAutoencoder(
    input_dim=768,
    hidden_dim=2048
).to(device)

sae.load_state_dict(
    torch.load(
        SAE_PATH,
        map_location=device
    )
)

sae.eval()

with open(
    CLASSIFIER_PATH,
    "rb"
) as file:

    classifier = pickle.load(
        file
    )

probabilities = []

batch_size = 32

for start in range(
    0,
    len(texts),
    batch_size
):

    batch = texts[
        start:start + batch_size
    ]

    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = encoder(
            **inputs
        )

        hidden_states = (
            outputs.last_hidden_state
        )

        attention_mask = (
            inputs["attention_mask"]
        )

        mask = attention_mask.unsqueeze(
            -1
        ).expand(
            hidden_states.size()
        ).float()

        masked_hidden_states = (
            hidden_states * mask
        )

        pooled = (
            masked_hidden_states.sum(
                dim=1
            )
            / mask.sum(
                dim=1
            )
        )

        _, sparse_features = sae(
            pooled
        )

        batch_probabilities = (
            classifier.predict_proba(
                sparse_features.cpu().numpy()
            )[:, 1]
        )

    probabilities.extend(
        batch_probabilities
    )


probabilities = np.array(
    probabilities
)

print(
    "OASST1 benign validation examples:",
    len(probabilities)
)

print(
    "\nFalse-positive analysis"
)

print(
    "=" * 60
)

for threshold in np.arange(
    0.50,
    0.96,
    0.05
):

    false_positives = np.sum(
        probabilities >= threshold
    )

    false_positive_rate = (
        false_positives /
        len(probabilities)
    )

    print(
        f"Threshold: {threshold:.2f}"
    )

    print(
        f"False positives: "
        f"{false_positives} / {len(probabilities)}"
    )

    print(
        f"False positive rate: "
        f"{false_positive_rate:.4f}"
    )

    print()