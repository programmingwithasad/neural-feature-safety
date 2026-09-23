from datasets import load_dataset


dataset = load_dataset(
    "OpenAssistant/oasst1"
)

train = dataset["train"]

scores = {
    "toxicity": [],
    "severe_toxicity": [],
    "obscene": [],
    "identity_attack": [],
    "insult": [],
    "threat": [],
    "sexual_explicit": []
}

count = 0

for row in train:

    if row["lang"] != "en":
        continue

    if row["role"] != "prompter":
        continue

    if row["deleted"]:
        continue

    if row["review_result"] is not True:
        continue

    detoxify = row["detoxify"]

    for key in scores:

        value = detoxify.get(key)

        if value is not None:
            scores[key].append(value)

    count += 1

print(
    "Filtered English prompter messages:",
    count
)

print(
    "\nModeration statistics"
)

for key, values in scores.items():

    if not values:
        continue

    values.sort()

    middle = len(values) // 2

    if len(values) % 2 == 0:

        median = (
            values[middle - 1] +
            values[middle]
        ) / 2

    else:

        median = values[middle]

    print(
        f"\n{key}"
    )

    print(
        f"Mean: {sum(values) / len(values):.6f}"
    )

    print(
        f"Median: {median:.6f}"
    )

    print(
        f"Maximum: {max(values):.6f}"
    )

    print(
        f"Above 0.10: "
        f"{sum(v > 0.10 for v in values)}"
    )

    print(
        f"Above 0.20: "
        f"{sum(v > 0.20 for v in values)}"
    )

    print(
        f"Above 0.50: "
        f"{sum(v > 0.50 for v in values)}"
    )