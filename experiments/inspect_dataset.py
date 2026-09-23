import pandas as pd

train_path = "data/raw/wildguard_train.parquet"
test_path = "data/raw/wildguard_test.parquet"

train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)

print("TRAIN SHAPE:", train_data.shape)
print("TEST SHAPE:", test_data.shape)

print("\nPROMPT HARM LABELS")
print(train_data["prompt_harm_label"].value_counts(dropna=False))

print("\nRESPONSE REFUSAL LABELS")
print(train_data["response_refusal_label"].value_counts(dropna=False))

print("\nRESPONSE HARM LABELS")
print(train_data["response_harm_label"].value_counts(dropna=False))

print("\nSUBCATEGORIES")
print(train_data["subcategory"].value_counts(dropna=False))

print("\nADVERSARIAL")
print(train_data["adversarial"].value_counts(dropna=False))