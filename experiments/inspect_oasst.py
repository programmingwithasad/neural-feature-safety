from datasets import load_dataset


dataset = load_dataset(
    "OpenAssistant/oasst1"
)

train = dataset["train"]

print("Total training messages:", len(train))

print("\nLanguages")

languages = {}

for value in train["lang"]:

    languages[value] = (
        languages.get(value, 0) + 1
    )

for language, count in sorted(
    languages.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        language,
        count
    )

print("\nRoles")

roles = {}

for value in train["role"]:

    roles[value] = (
        roles.get(value, 0) + 1
    )

for role, count in roles.items():

    print(
        role,
        count
    )

print("\nSynthetic")

synthetic = {}

for value in train["synthetic"]:

    synthetic[value] = (
        synthetic.get(value, 0) + 1
    )

for value, count in synthetic.items():

    print(
        value,
        count
    )

print("\nDeleted")

deleted = {}

for value in train["deleted"]:

    deleted[value] = (
        deleted.get(value, 0) + 1
    )

for value, count in deleted.items():

    print(
        value,
        count
    )

print("\nReview results")

review = {}

for value in train["review_result"]:

    review[str(value)] = (
        review.get(str(value), 0) + 1
    )

for value, count in review.items():

    print(
        value,
        count
    )
    