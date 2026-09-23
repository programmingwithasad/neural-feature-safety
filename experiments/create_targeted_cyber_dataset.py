import pandas as pd


WILDGUARD_PATH = "data/processed/prompt_train.csv"
OASST_PATH = "data/processed/oasst_targeted_clean.csv"
OUTPUT_PATH = "data/processed/targeted_cyber_dataset.csv"

TARGET_PER_GROUP = 200


wildguard = pd.read_csv(
    WILDGUARD_PATH
)

oasst = pd.read_csv(
    OASST_PATH
)


wildguard_harmful = wildguard[
    wildguard["subcategory"].eq("cyberattack")
    & wildguard["prompt_harm_label"].eq("harmful")
][
    ["prompt"]
].copy()

wildguard_harmful["text"] = (
    wildguard_harmful["prompt"]
    .astype(str)
    .str.strip()
)

wildguard_harmful["label"] = "harmful"
wildguard_harmful["target_category"] = "harmful_cyber"


oasst_humor = oasst[
    oasst["target_category"].eq("benign_humor")
][
    ["text"]
].copy()

oasst_humor["label"] = "unharmful"
oasst_humor["target_category"] = "benign_humor"


oasst_cyber = oasst[
    oasst["target_category"].eq("benign_cyber")
][
    ["text"]
].copy()

oasst_cyber["label"] = "unharmful"
oasst_cyber["target_category"] = "benign_cyber"


wildguard_harmful = (
    wildguard_harmful
    .drop_duplicates("text")
    .sample(
        n=min(
            TARGET_PER_GROUP,
            len(wildguard_harmful)
        ),
        random_state=42
    )
)

oasst_humor = (
    oasst_humor
    .drop_duplicates("text")
    .sample(
        n=min(
            TARGET_PER_GROUP,
            len(oasst_humor)
        ),
        random_state=42
    )
)

oasst_cyber = (
    oasst_cyber
    .drop_duplicates("text")
    .sample(
        n=min(
            TARGET_PER_GROUP,
            len(oasst_cyber)
        ),
        random_state=42
    )
)


columns = [
    "text",
    "label",
    "target_category"
]


targeted = pd.concat(
    [
        oasst_humor[columns],
        oasst_cyber[columns],
        wildguard_harmful[columns]
    ],
    ignore_index=True
)


targeted = (
    targeted
    .drop_duplicates("text")
    .sample(
        frac=1,
        random_state=42
    )
    .reset_index(drop=True)
)


targeted.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "Targeted cyber dataset created."
)

print(
    f"Total examples: {len(targeted)}"
)

print()

print(
    "Category distribution:"
)

print(
    targeted[
        "target_category"
    ].value_counts()
)

print()

print(
    "Label distribution:"
)

print(
    targeted[
        "label"
    ].value_counts()
)

print()

print(
    f"Saved to: {OUTPUT_PATH}"
)