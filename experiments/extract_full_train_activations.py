import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/processed/prompt_train.csv"
OUTPUT_PATH = "data/processed/prompt_train_activations_full.pt"

BATCH_SIZE = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

data = pd.read_csv(DATA_PATH)

data = data.dropna(
    subset=["prompt", "prompt_harm_label"]
)

data = data[
    data["prompt"].astype(str).str.strip() != ""
]

texts = data["prompt"].astype(str).tolist()

labels = data["prompt_harm_label"].tolist()

print("Number of training samples:", len(texts))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(device)

model.eval()

all_features = []

for start in range(
    0,
    len(texts),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(texts)
    )

    batch_texts = texts[start:end]

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

    with torch.no_grad():

        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state

    attention_mask = inputs["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(
        hidden_states.size()
    ).float()

    masked_hidden_states = (
        hidden_states * mask
    )

    summed = masked_hidden_states.sum(
        dim=1
    )

    counts = mask.sum(dim=1)

    pooled = summed / counts

    all_features.append(
        pooled.cpu()
    )

    print(
        "Processed:",
        end,
        "/",
        len(texts)
    )

features = torch.cat(
    all_features
)

labels_tensor = torch.tensor(
    [
        1 if label == "harmful" else 0
        for label in labels
    ]
)

output = {
    "features": features,
    "labels": labels_tensor,
    "texts": texts
}

Path(
    "data/processed"
).mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    output,
    OUTPUT_PATH
)

print("\nExtraction completed.")

print(
    "Feature shape:",
    features.shape
)

print(
    "Labels shape:",
    labels_tensor.shape
)

print(
    "Saved to:",
    OUTPUT_PATH
)