import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


TRAIN_PATH = "data/raw/wildguard_train.parquet"
TEST_PATH = "data/raw/wildguard_test.parquet"

OUTPUT_DIR = Path("data/processed")


def prepare_prompt_data(data):
    prompt_data = data[
        [
            "prompt",
            "prompt_harm_label",
            "adversarial",
            "subcategory",
        ]
    ].copy()

    prompt_data = prompt_data.dropna(
        subset=["prompt", "prompt_harm_label"]
    )

    prompt_data = prompt_data[
        prompt_data["prompt"].astype(str).str.strip() != ""
    ]

    return prompt_data


def prepare_response_data(data):
    response_data = data[
        [
            "response",
            "response_harm_label",
            "response_refusal_label",
            "subcategory",
        ]
    ].copy()

    response_data = response_data[
        response_data["response_harm_label"].isin(
            ["harmful", "unharmful"]
        )
    ]

    response_data = response_data.dropna(
        subset=["response"]
    )

    response_data = response_data[
        response_data["response"].astype(str).str.strip() != ""
    ]

    return response_data


def split_training_data(data, label_column):
    train_data, validation_data = train_test_split(
        data,
        test_size=0.10,
        random_state=42,
        stratify=data[label_column]
    )

    return train_data, validation_data


def main():

    print("Loading WildGuardMix dataset...")

    train_data = pd.read_parquet(TRAIN_PATH)
    test_data = pd.read_parquet(TEST_PATH)

    prompt_data = prepare_prompt_data(train_data)
    prompt_test = prepare_prompt_data(test_data)

    response_data = prepare_response_data(train_data)
    response_test = prepare_response_data(test_data)

    prompt_train, prompt_validation = split_training_data(
        prompt_data,
        "prompt_harm_label"
    )

    response_train, response_validation = split_training_data(
        response_data,
        "response_harm_label"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt_train.to_csv(
        OUTPUT_DIR / "prompt_train.csv",
        index=False
    )

    prompt_validation.to_csv(
        OUTPUT_DIR / "prompt_validation.csv",
        index=False
    )

    prompt_test.to_csv(
        OUTPUT_DIR / "prompt_test.csv",
        index=False
    )

    response_train.to_csv(
        OUTPUT_DIR / "response_train.csv",
        index=False
    )

    response_validation.to_csv(
        OUTPUT_DIR / "response_validation.csv",
        index=False
    )

    response_test.to_csv(
        OUTPUT_DIR / "response_test.csv",
        index=False
    )

    print("\nPrompt datasets:")
    print("Train:", len(prompt_train))
    print("Validation:", len(prompt_validation))
    print("Test:", len(prompt_test))

    print("\nResponse datasets:")
    print("Train:", len(response_train))
    print("Validation:", len(response_validation))
    print("Test:", len(response_test))

    print("\nPrompt training labels:")
    print(prompt_train["prompt_harm_label"].value_counts())

    print("\nPrompt validation labels:")
    print(prompt_validation["prompt_harm_label"].value_counts())

    print("\nResponse training labels:")
    print(response_train["response_harm_label"].value_counts())

    print("\nResponse validation labels:")
    print(response_validation["response_harm_label"].value_counts())


if __name__ == "__main__":
    main()