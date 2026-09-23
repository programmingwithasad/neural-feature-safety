import torch
import pickle
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel

from autoencoder import SparseAutoencoder


DATA_PATH = "data/processed/oasst_benign_validation.csv"
SAE_PATH = "models/sparse_autoencoder_full.pt"
CLASSIFIER_PATH = "models/combined_safety_classifier.pkl"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = pd.read_csv(
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

results = []

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

        probabilities = classifier.predict_proba(
            sparse_features.cpu().numpy()
        )[:, 1]

    for text, probability in zip(
        batch,
        probabilities
    ):

        results.append(
            {
                "text": text,
                "harmful_probability": float(
                    probability
                )
            }
        )


results = sorted(
    results,
    key=lambda x: x["harmful_probability"],
    reverse=True
)

print(
    "Highest-scoring benign OASST1 examples"
)

print(
    "=" * 80
)

for index, result in enumerate(
    results[:50],
    start=1
):

    print(
        f"\nRank: {index}"
    )

    print(
        f"Harmful probability: "
        f"{result['harmful_probability']:.4f}"
    )

    print(
        f"Text: {result['text']}"
    )
    