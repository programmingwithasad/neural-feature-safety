import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModel


project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.autoencoder import SparseAutoencoder

benchmark_path = os.path.join(
    project_root,
    "data",
    "processed",
    "independent_gateway_benchmark.json"
)

output_path = os.path.join(
    project_root,
    "data",
    "processed",
    "independent_gateway_benchmark_features.pt"
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


with open(
    benchmark_path,
    "r",
    encoding="utf-8"
) as f:
    benchmark = json.load(f)


examples = benchmark["examples"]

texts = [
    example["text"]
    for example in examples
]


print("=" * 80)
print("EXTRACTING INDEPENDENT GATEWAY FEATURES")
print("=" * 80)

print(f"\nExamples: {len(texts)}")
print(f"Device: {device}")


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

sae_path = os.path.join(
    project_root,
    "models",
    "sparse_autoencoder_full.pt"
)

checkpoint = torch.load(
    sae_path,
    map_location=device
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    sae.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    sae.load_state_dict(checkpoint)

sae = sae.to(device)
sae.eval()


all_features = []


with torch.no_grad():

    for start in range(
        0,
        len(texts),
        16
    ):

        batch_texts = texts[
            start:start + 16
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

        mask = inputs[
            "attention_mask"
        ].unsqueeze(-1)

        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        _, sparse_features = sae(
            pooled
        )

        all_features.append(
            sparse_features.cpu()
        )

        print(
            f"Processed "
            f"{min(start + 16, len(texts))}/"
            f"{len(texts)}"
        )


features = torch.cat(
    all_features,
    dim=0
)


labels = torch.tensor([
    1 if example["expected"] == "BLOCK"
    else 0
    for example in examples
])


result = {
    "features": features,
    "labels": labels,
    "texts": texts,
    "metadata": examples
}


torch.save(
    result,
    output_path
)


print("\n" + "=" * 80)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 80)

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

print("\nSaved:")
print(output_path)