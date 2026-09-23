import torch
import joblib

from transformers import AutoTokenizer, AutoModel

from autoencoder import SparseAutoencoder


MODEL_NAME = "distilbert-base-uncased"

SAE_PATH = "models/sparse_autoencoder_full.pt"

CLASSIFIER_PATH = (
    "models/final_conversation_weighted_classifier.pkl"
)

BLOCK_THRESHOLD = 0.50
REVIEW_THRESHOLD = 0.30


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("Loading safety controller...")
print("Using device:", device)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


distilbert = AutoModel.from_pretrained(
    MODEL_NAME
).to(device)

distilbert.eval()


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


classifier = joblib.load(
    CLASSIFIER_PATH
)


print("Safety controller loaded.")


def analyze(text):

    inputs = tokenizer(
        [text],
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

        outputs = distilbert(
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

        _, sparse_features = sae(
            pooled
        )


    sparse_features = (
        sparse_features.detach()
        .cpu()
        .numpy()
    )


    probabilities = classifier.predict_proba(
        sparse_features
    )[0]


    unharmful_probability = probabilities[0]

    harmful_probability = probabilities[1]


    if harmful_probability >= BLOCK_THRESHOLD:

        risk_level = "HIGH RISK"

        decision = "BLOCK"

    elif harmful_probability >= REVIEW_THRESHOLD:

        risk_level = "MEDIUM RISK"

        decision = "REVIEW"

    else:

        risk_level = "LOW RISK"

        decision = "ALLOW"


    return {
        "harmful_probability":
            harmful_probability,

        "unharmful_probability":
            unharmful_probability,

        "risk_level":
            risk_level,

        "decision":
            decision
    }


print()
print(
    "========================================"
)

print(
    "      NEURAL FEATURE SAFETY CONTROLLER"
)

print(
    "========================================"
)

print(
    f"Block threshold: {BLOCK_THRESHOLD:.2f}"
)

print(
    f"Review threshold: {REVIEW_THRESHOLD:.2f}"
)

print(
    "Type 'exit' to stop."
)

print()


while True:

    text = input(
        "Enter prompt: "
    )


    if text.lower() == "exit":

        break


    if not text.strip():

        print(
            "Please enter a prompt.\n"
        )

        continue


    result = analyze(
        text
    )


    print()
    print(
        "Safety Analysis"
    )

    print(
        "----------------"
    )

    print(
        f"Harmful probability: "
        f"{result['harmful_probability']:.4f}"
    )

    print(
        f"Unharmful probability: "
        f"{result['unharmful_probability']:.4f}"
    )

    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )

    print(
        f"Decision: "
        f"{result['decision']}"
    )


    if result["decision"] == "BLOCK":

        print()
        print(
            "User prompt blocked."
        )

        print(
            "The request was not passed to the AI model."
        )


    elif result["decision"] == "REVIEW":

        print()
        print(
            "User prompt requires review."
        )


    else:

        print()
        print(
            "User prompt allowed."
        )


    print(
        "\n----------------------------------------\n"
    )
