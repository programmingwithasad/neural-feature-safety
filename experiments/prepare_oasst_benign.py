from datasets import load_dataset
import pandas as pd


print("Loading OpenAssistant OASST1...")

dataset = load_dataset(
    "OpenAssistant/oasst1"
)

train = dataset["train"]
validation = dataset["validation"]


def is_benign(row):

    if row["lang"] != "en":
        return False

    if row["role"] != "prompter":
        return False

    if row["deleted"]:
        return False

    if row["review_result"] is not True:
        return False

    detoxify = row["detoxify"]

    thresholds = [
        detoxify.get("toxicity", 1.0),
        detoxify.get("severe_toxicity", 1.0),
        detoxify.get("obscene", 1.0),
        detoxify.get("identity_attack", 1.0),
        detoxify.get("insult", 1.0),
        detoxify.get("threat", 1.0),
        detoxify.get("sexual_explicit", 1.0)
    ]

    if any(
        value >= 0.10
        for value in thresholds
    ):
        return False

    return True


print("Filtering training messages...")

train_rows = []

for row in train:

    if is_benign(row):

        train_rows.append(
            {
                "text": row["text"],
                "label": "unharmful"
            }
        )


print("Filtering validation messages...")

validation_rows = []

for row in validation:

    if is_benign(row):

        validation_rows.append(
            {
                "text": row["text"],
                "label": "unharmful"
            }
        )


train_df = pd.DataFrame(
    train_rows
)

validation_df = pd.DataFrame(
    validation_rows
)

train_df = train_df.drop_duplicates(
    subset=["text"]
)

validation_df = validation_df.drop_duplicates(
    subset=["text"]
)

train_df.to_csv(
    "data/processed/oasst_benign_train.csv",
    index=False
)

validation_df.to_csv(
    "data/processed/oasst_benign_validation.csv",
    index=False
)

print(
    "\nOASST1 preparation completed."
)

print(
    f"Training examples: {len(train_df)}"
)

print(
    f"Validation examples: {len(validation_df)}"
)

print(
    "\nTraining labels:"
)

print(
    train_df["label"].value_counts()
)

print(
    "\nValidation labels:"
)

print(
    validation_df["label"].value_counts()
)