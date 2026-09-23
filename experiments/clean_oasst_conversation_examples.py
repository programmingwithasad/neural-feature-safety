import pandas as pd
import re


INPUT_PATH = "data/processed/oasst_conversation_examples.csv"
OUTPUT_PATH = "data/processed/oasst_conversation_clean.csv"

df = pd.read_csv(INPUT_PATH)

df["text"] = df["text"].fillna("").astype(str)

patterns = [
    r"^\s*(hi|hello|hey)\s*[!.?,]?\s*$",
    r"^\s*(hi|hello|hey)\s+(there|everyone)\s*[!.?,]?\s*$",
    r"^\s*(hi|hello|hey)[, ]+how are you(?: doing)?[!?.,]?\s*$",
    r"^\s*how are you(?: doing)?[!?.,]?\s*$",
    r"^\s*how's your day[!?.,]?\s*$",
    r"^\s*how have you been[!?.,]?\s*$",
    r"^\s*(what's|what is) up[!?.,]?\s*$",
    r"^\s*(good morning|good afternoon|good evening)[!.]?\s*$",
    r"^\s*(thanks|thank you)[!.]?\s*$",
    r"^\s*(you're welcome|you are welcome)[!.]?\s*$",
    r"^\s*(goodbye|bye|see you later)[!.]?\s*$",
    r"^\s*(nice to meet you)[!.]?\s*$",
    r"^\s*(tell me a joke)[!.?]?\s*$",
    r"^\s*(tell me something funny)[!.?]?\s*$",
    r"^\s*(make me laugh)[!.?]?\s*$",
    r"^\s*(can you tell me a joke)[!.?]?\s*$",
    r"^\s*(do you know any jokes)[!.?]?\s*$",
    r"^\s*(what's new)[!.?]?\s*$",
    r"^\s*(what's going on)[!.?]?\s*$"
]

regex = re.compile(
    "|".join(patterns),
    flags=re.IGNORECASE
)

clean = df[
    df["text"].str.match(
        regex,
        na=False
    )
].copy()

clean = clean.drop_duplicates(
    subset=["text"]
)

clean["label"] = "unharmful"

clean = clean[
    ["text", "label"]
]

clean.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Clean conversational dataset created.")
print(f"Original mined examples: {len(df)}")
print(f"Clean conversational examples: {len(clean)}")
print(f"Saved to: {OUTPUT_PATH}")

print()
print("Examples:")
print("=" * 70)

for i, text in enumerate(
    clean["text"].head(50),
    start=1
):
    print(f"{i}. {text}")