import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModel


INPUT_PATH = "data/processed/oasst_benign_train.csv"
OUTPUT_PATH = "data/processed/oasst_benign_activations.pt"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    "Using device:",
    device
)

data = pd.read_csv(
    INPUT_PATH
)

texts = data["text"].fillna("").tolist()

labels = torch.zeros(
    len(texts),
    dtype=torch.long
)

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

model = AutoModel.from_pretrained(
    "distilbert-base-uncased"
).to(device)

model.eval()

features = []

batch_size = 16

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

        outputs = model(
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

        summed = masked_hidden_states.sum(
            dim=1
        )

        counts = mask.sum(
            dim=1
        )

        pooled = summed / counts

        features.append(
            pooled.cpu()
        )

    processed = min(
        start + batch_size,
        len(texts)
    )

    if processed % 1000 == 0 or processed == len(texts):

        print(
            f"Processed: {processed} / {len(texts)}"
        )


features = torch.cat(
    features
)

torch.save(
    {
        "features": features,
        "labels": labels
    },
    OUTPUT_PATH
)

print(
    "\nExtraction completed."
)

print(
    "Feature shape:",
    features.shape
)

print(
    "Labels shape:",
    labels.shape
)

print(
    "Saved to:",
    OUTPUT_PATH
)
