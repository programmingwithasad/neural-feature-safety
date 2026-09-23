import pandas as pd


INPUT_PATH = "data/processed/oasst_hard_negatives.csv"
OUTPUT_PATH = "data/processed/oasst_curated_hard_negatives.csv"


df = pd.read_csv(INPUT_PATH)

# Conservative selection:
# These are OASST examples already labeled unharmful,
# but our classifier currently gives them a high harmful score.
#
# We use a probability range rather than blindly taking
# every high-scoring example. Extremely high-scoring examples
# can contain genuinely harmful or ambiguous requests.

curated = df[
    (df["harmful_probability"] >= 0.50)
    & (df["harmful_probability"] < 0.95)
].copy()

# OASST examples used here are already labeled unharmful.
curated["label"] = "unharmful"

# Remove accidental duplicates just in case.
curated = curated.drop_duplicates(
    subset=["text"]
)

# Keep the most informative columns.
curated = curated[
    [
        "text",
        "label",
        "harmful_probability"
    ]
]

curated = curated.sort_values(
    "harmful_probability",
    ascending=False
)

curated.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Curated hard-negative preparation completed.")
print(f"Original mined examples: {len(df)}")
print(f"Curated hard negatives: {len(curated)}")
print()
print("Probability distribution:")
print(
    curated["harmful_probability"]
    .describe()
)
print()
print(f"Saved to: {OUTPUT_PATH}")