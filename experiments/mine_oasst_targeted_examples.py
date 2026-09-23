import ast
import pandas as pd
from datasets import load_dataset


OUTPUT_PATH = "data/processed/oasst_targeted_examples.csv"

dataset = load_dataset(
    "OpenAssistant/oasst1",
    split="train"
)

rows = []

for item in dataset:

    if item["role"] != "prompter":
        continue

    if item["lang"] != "en":
        continue

    if item["deleted"]:
        continue

    text = str(
        item["text"]
    ).strip()

    if not text:
        continue

    labels = item["labels"]

    if isinstance(labels, str):
        try:
            labels = ast.literal_eval(
                labels
            )
        except Exception:
            continue

    if not isinstance(labels, dict):
        continue

    names = labels.get(
        "name",
        []
    )

    values = labels.get(
        "value",
        []
    )

    label_map = dict(
        zip(
            names,
            values
        )
    )

    toxicity = float(
        label_map.get(
            "toxicity",
            0.0
        )
    )

    violence = float(
        label_map.get(
            "violence",
            0.0
        )
    )

    not_appropriate = float(
        label_map.get(
            "not_appropriate",
            0.0
        )
    )

    text_lower = text.lower()

    joke_terms = [
        "joke",
        "funny",
        "humor",
        "humour",
        "laugh",
        "hilarious",
        "comedy",
        "make me laugh"
    ]

    cyber_terms = [
        "password",
        "account",
        "cyber",
        "cybersecurity",
        "security",
        "hacking",
        "hack",
        "malware",
        "phishing",
        "authentication",
        "credentials",
        "login",
        "privacy",
        "data breach"
    ]

    harmful_cyber_terms = [
        "steal",
        "break into",
        "bypass",
        "crack",
        "compromise",
        "access someone",
        "access another",
        "take over",
        "credential",
        "password"
    ]

    is_joke = any(
        term in text_lower
        for term in joke_terms
    )

    is_cyber = any(
        term in text_lower
        for term in cyber_terms
    )

    is_harmful_cyber = (
        is_cyber
        and any(
            term in text_lower
            for term in harmful_cyber_terms
        )
    )

    if is_joke:

        if (
            toxicity <= 0.05
            and violence <= 0.05
            and not_appropriate <= 0.05
        ):

            rows.append(
                {
                    "text": text,
                    "label": "unharmful",
                    "target_category": "benign_humor",
                    "toxicity": toxicity,
                    "violence": violence,
                    "not_appropriate": not_appropriate
                }
            )

    elif is_cyber:

        if is_harmful_cyber:

            if (
                toxicity <= 0.50
                and violence <= 0.50
                and not_appropriate <= 0.50
            ):

                rows.append(
                    {
                        "text": text,
                        "label": "harmful",
                        "target_category": "cyber_intent",
                        "toxicity": toxicity,
                        "violence": violence,
                        "not_appropriate": not_appropriate
                    }
                )

        else:

            if (
                toxicity <= 0.05
                and violence <= 0.05
                and not_appropriate <= 0.05
            ):

                rows.append(
                    {
                        "text": text,
                        "label": "unharmful",
                        "target_category": "benign_cyber",
                        "toxicity": toxicity,
                        "violence": violence,
                        "not_appropriate": not_appropriate
                    }
                )


df = pd.DataFrame(
    rows
)

if len(df) > 0:

    df = df.drop_duplicates(
        subset=["text"]
    ).reset_index(
        drop=True
    )

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "Targeted OASST mining completed."
)

print(
    f"Total targeted examples: {len(df)}"
)

print()

if len(df) > 0:

    print(
        "Category distribution:"
    )

    print(
        df["target_category"].value_counts()
    )

    print()

    print(
        "Label distribution:"
    )

    print(
        df["label"].value_counts()
    )

    print()

    print(
        "Examples:"
    )

    print(
        "=" * 70
    )

    for i, row in df.head(
        30
    ).iterrows():

        print(
            f"{i + 1}. "
            f"[{row['target_category']}] "
            f"{row['text']}"
        )

print()

print(
    f"Saved to: {OUTPUT_PATH}"
)