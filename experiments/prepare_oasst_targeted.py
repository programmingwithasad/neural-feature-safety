import pandas as pd

INPUT_PATH = "data/processed/oasst_targeted_examples.csv"
OUTPUT_PATH = "data/processed/oasst_targeted_clean.csv"

df = pd.read_csv(INPUT_PATH)

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["text"].str.len() >= 5]

df = df.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)

harmful_patterns = [
    "steal someone's password",
    "steal someone else's password",
    "steal passwords",
    "steal credentials",
    "break into someone's account",
    "break into another person's account",
    "break into an account",
    "bypass authentication",
    "bypass someone's authentication",
    "crack someone's password",
    "crack a password",
    "take over someone's account",
    "take over an account",
    "access another person's account",
    "access someone else's account",
    "access another user's account",
    "compromise someone's account",
    "compromise an account",
    "phishing attack",
    "credential theft"
]

benign_patterns = [
    "password manager",
    "choose a password",
    "strong password",
    "password policy",
    "security principles",
    "network security",
    "network performance",
    "security best practices",
    "cybersecurity",
    "cyber security",
    "security vulnerabilities",
    "wireshark",
    "sudo",
    "systemctl",
    "docker",
    "mysql",
    "mysqli",
    "html",
    "css",
    "javascript",
    "authentication step",
    "authentication system",
    "privacy policy"
]

def contains_pattern(text, patterns):
    text_lower = text.lower()
    return any(
        pattern in text_lower
        for pattern in patterns
    )

def assign_label(row):
    text = row["text"]

    if row["target_category"] == "benign_humor":
        return "unharmful"

    if row["target_category"] == "benign_cyber":
        return "unharmful"

    if row["target_category"] == "cyber_intent":

        if contains_pattern(
            text,
            harmful_patterns
        ):
            return "harmful"

        if contains_pattern(
            text,
            benign_patterns
        ):
            return "unharmful"

        return "review"

    return "review"

df["label"] = df.apply(
    assign_label,
    axis=1
)

review_df = df[
    df["label"] == "review"
].copy()

clean_df = df[
    df["label"].isin(
        ["harmful", "unharmful"]
    )
].copy()

clean_df = clean_df[
    [
        "text",
        "label",
        "target_category",
        "toxicity",
        "violence",
        "not_appropriate"
    ]
]

review_path = (
    "data/processed/"
    "oasst_targeted_review.csv"
)

review_df.to_csv(
    review_path,
    index=False
)

clean_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "Targeted OASST preparation completed."
)

print(
    f"Original examples: {len(df)}"
)

print(
    f"Clean training examples: {len(clean_df)}"
)

print(
    f"Review examples: {len(review_df)}"
)

print()

print(
    "Training label distribution:"
)

print(
    clean_df["label"].value_counts()
)

print()

print(
    "Training category distribution:"
)

print(
    clean_df["target_category"].value_counts()
)

print()

print(
    f"Saved clean dataset to: {OUTPUT_PATH}"
)

print(
    f"Saved review dataset to: {review_path}"
)

print()

if len(review_df) > 0:

    print(
        "Examples requiring review:"
    )

    print(
        "=" * 70
    )

    for i, row in review_df.iterrows():

        print(
            f"{i + 1}. "
            f"{row['text']}"
        )