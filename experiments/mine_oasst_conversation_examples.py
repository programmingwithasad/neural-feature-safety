import pandas as pd
from datasets import load_dataset


OUTPUT_PATH = "data/processed/oasst_conversation_examples.csv"

dataset = load_dataset(
    "OpenAssistant/oasst1",
    split="train"
)

df = dataset.to_pandas()

df = df[
    (df["lang"] == "en") &
    (df["role"] == "prompter") &
    (df["deleted"] == False)
].copy()

df["text"] = df["text"].fillna("").astype(str)

patterns = [
    "hi",
    "hello",
    "hey",
    "how are you",
    "how's your day",
    "how have you been",
    "what's up",
    "what is up",
    "good morning",
    "good afternoon",
    "good evening",
    "thank",
    "thanks",
    "you're welcome",
    "tell me a joke",
    "tell me something funny",
    "make me laugh",
    "funny",
    "joke",
    "can you help me",
    "nice to meet you",
    "goodbye",
    "bye",
    "see you"
]

pattern = "|".join(
    [p.replace("'", "['’]") for p in patterns]
)

matches = df[
    df["text"].str.lower().str.contains(
        pattern,
        regex=True,
        na=False
    )
].copy()

matches = matches.drop_duplicates(
    subset=["text"]
)

matches["label"] = "unharmful"

matches = matches[
    ["text", "label"]
]

matches.to_csv(
    OUTPUT_PATH,
    index=False
)

print("OASST conversation mining completed.")
print(f"Total English prompter messages: {len(df)}")
print(f"Conversation matches: {len(matches)}")
print(f"Saved to: {OUTPUT_PATH}")

print()
print("Sample conversation examples:")
print("=" * 70)

for i, text in enumerate(
    matches["text"].head(30),
    start=1
):
    print()
    print(f"{i}. {text}")