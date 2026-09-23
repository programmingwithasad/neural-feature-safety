import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "distilbert-base-uncased"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

text = "Explain the basic concepts of machine learning."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.last_hidden_state

print("Input text:")
print(text)

print("\nTokenized input shape:")
print(inputs["input_ids"].shape)

print("\nHidden state shape:")
print(hidden_states.shape)

print("\nDevice:")
print(hidden_states.device)

print("\nFirst token representation:")
print(hidden_states[0, 0, :10])