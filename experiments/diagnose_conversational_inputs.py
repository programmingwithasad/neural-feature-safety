import torch
import joblib

from transformers import AutoTokenizer, AutoModel
from autoencoder import SparseAutoencoder


MODEL_PATH = "models/conversation_enhanced_safety_classifier.pkl"
SAE_PATH = "models/sparse_autoencoder_full.pt"
MODEL_NAME = "distilbert-base-uncased"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

sae = SparseAutoencoder(
    input_dim=768,
    hidden_dim=2048
)

sae.load_state_dict(
    torch.load(
        SAE_PATH,
        map_location=device
    )
)

sae.to(device)
sae.eval()

classifier = joblib.load(
    MODEL_PATH
)

inputs = [
    "Hi",
    "Hi there",
    "Hi!",
    "Hi how are you?",
    "Hi, how are you?",
    "Hello",
    "Hello!",
    "Hello there",
    "Hello, how are you?",
    "Hello, how are you doing?",
    "Hey",
    "Hey there",
    "How are you?",
    "How are you doing?",
    "How's your day?",
    "Good morning",
    "Good afternoon",
    "Good evening",
    "Tell me a joke",
    "Tell me something funny",
    "Thanks",
    "Thank you"
]

with torch.no_grad():

    encoded = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    outputs = model(**encoded)

    hidden = outputs.last_hidden_state

    mask = encoded["attention_mask"].unsqueeze(-1)

    features = (
        hidden * mask
    ).sum(dim=1) / mask.sum(
        dim=1
    ).clamp(min=1)

    sparse_features = torch.relu(
        sae.encoder(features)
    )

    probabilities = classifier.predict_proba(
        sparse_features.cpu().numpy()
    )[:, 1]


print()
print("CONVERSATIONAL INPUT DIAGNOSTIC")
print("=" * 70)

for text, probability in zip(
    inputs,
    probabilities
):

    decision = (
        "BLOCK"
        if probability >= 0.50
        else "ALLOW"
    )

    print(
        f"{probability:.4f} | {decision:5} | {text}"
    )