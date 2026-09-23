from datasets import load_dataset


print("Loading OpenAssistant OASST1 dataset...")

dataset = load_dataset(
    "OpenAssistant/oasst1"
)

print("\nDataset loaded.")

print(
    f"Train rows: {len(dataset['train'])}"
)

print(
    f"Validation rows: {len(dataset['validation'])}"
)

print(
    "\nColumns:"
)

print(
    dataset["train"].column_names
)

print(
    "\nFirst example:"
)

print(
    dataset["train"][0]
)

print(
    "\nLanguages:"
)

print(
    dataset["train"]["lang"][:20]
)

print(
    "\nRoles:"
)

print(
    set(dataset["train"]["role"])
)