import json
import os
import sys

import torch

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import AutoTokenizer, AutoModel
from src.autoencoder import SparseAutoencoder


dataset_path = os.path.join(
    project_root,
    "data",
    "processed",
    "gateway_safety_benchmark.json"
)

output_path = os.path.join(
    project_root,
    "data",
    "processed",
    "gateway_safety_benchmark_features.pt"
)

sae_path = os.path.join(
    project_root,
    "models",
    "sparse_autoencoder_full.pt"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("GATEWAY BENCHMARK SAE FEATURE EXTRACTION")
print("=" * 70)

print(f"Device: {device}")


with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)


tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

encoder = AutoModel.from_pretrained(
    "distilbert-base-uncased"
)

encoder = encoder.to(device)
encoder.eval()


sae = SparseAutoencoder(
    input_dim=768,
    hidden_dim=2048
)

checkpoint = torch.load(
    sae_path,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    sae.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    sae.load_state_dict(checkpoint)

sae = sae.to(device)
sae.eval()


texts = [
    sample["text"]
    for sample in dataset
]

labels = torch.tensor(
    [
        1 if sample["expected"] == "BLOCK" else 0
        for sample in dataset
    ],
    dtype=torch.long
)


all_features = []

batch_size = 16

with torch.no_grad():

    for start in range(0, len(texts), batch_size):

        batch_texts = texts[
            start:start + batch_size
        ]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        outputs = encoder(
            **inputs
        )

        hidden = outputs.last_hidden_state

        attention_mask = inputs[
            "attention_mask"
        ].unsqueeze(-1)

        masked_hidden = (
            hidden * attention_mask
        )

        pooled = (
            masked_hidden.sum(dim=1)
            / attention_mask.sum(dim=1).clamp(
                min=1
            )
        )

        _, sparse_features = sae(
            pooled
        )

        all_features.append(
            sparse_features.cpu()
        )

        print(
            f"Processed "
            f"{min(start + batch_size, len(texts))}/"
            f"{len(texts)}"
        )


features = torch.cat(
    all_features,
    dim=0
)


output = {
    "features": features,
    "labels": labels,
    "categories": [
        sample["category"]
        for sample in dataset
    ],
    "topics": [
        sample["topic"]
        for sample in dataset
    ],
    "prompts": [
        sample["prompt"]
        for sample in dataset
    ],
    "texts": texts,
    "expected": [
        sample["expected"]
        for sample in dataset
    ]
}


os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)

torch.save(
    output,
    output_path
)


print("\n" + "=" * 70)
print("EXTRACTION COMPLETED")
print("=" * 70)

print(f"Samples: {len(texts)}")
print(f"Features: {features.shape}")
print(f"Labels: {labels.shape}")

print(f"\nSaved to:")
print(output_path)