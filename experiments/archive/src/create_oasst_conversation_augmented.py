import pandas as pd


INPUT_PATH = "data/processed/oasst_conversation_clean.csv"
OUTPUT_PATH = "data/processed/oasst_conversation_augmented.csv"


templates = [
    "Hi",
    "Hi there",
    "Hi!",
    "Hi, how are you?",
    "Hi how are you?",
    "Hi, how are you doing?",
    "Hi how are you doing?",
    "Hello",
    "Hello!",
    "Hello there",
    "Hello, how are you?",
    "Hello how are you?",
    "Hello, how are you doing?",
    "Hello how are you doing?",
    "Hey",
    "Hey!",
    "Hey there",
    "Hey, how are you?",
    "Hey how are you?",
    "Hey, how are you doing?",
    "How are you?",
    "How are you doing?",
    "How have you been?",
    "How is your day?",
    "How's your day?",
    "What's up?",
    "What is up?",
    "What's going on?",
    "Good morning",
    "Good morning!",
    "Good afternoon",
    "Good afternoon!",
    "Good evening",
    "Good evening!",
    "Nice to meet you.",
    "Nice to meet you!",
    "Thanks",
    "Thanks!",
    "Thanks a lot",
    "Thank you",
    "Thank you!",
    "Thank you very much",
    "I appreciate it",
    "That's helpful, thanks",
    "Thanks for your help",
    "You're welcome",
    "You are welcome",
    "Bye",
    "Bye!",
    "Goodbye",
    "Goodbye!",
    "See you later",
    "See you soon",
    "Tell me a joke",
    "Tell me a joke.",
    "Can you tell me a joke?",
    "Can you tell me a funny joke?",
    "Tell me something funny",
    "Tell me something funny.",
    "Can you make me laugh?",
    "Make me laugh",
    "Do you know any jokes?",
    "Do you know a funny joke?",
    "What's a good joke?",
    "Give me a joke",
    "Can you give me a joke?",
    "I want to hear a joke",
    "Say something funny",
    "Can you say something funny?",
    "I'm bored, tell me a joke",
    "I'm bored, tell me something funny",
    "Can you help me?",
    "Could you help me?",
    "Can you help me with something?",
    "I need some help",
    "Could you help me with this?",
    "What can you help me with?",
    "Can I ask you something?",
    "Can I ask you a question?",
    "I have a question",
    "I wanted to ask you something"
]


df = pd.read_csv(INPUT_PATH)

natural = df[
    ["text", "label"]
].copy()

augmented = pd.DataFrame(
    {
        "text": templates,
        "label": "unharmful"
    }
)

combined = pd.concat(
    [
        natural,
        augmented
    ],
    ignore_index=True
)

combined["text"] = combined[
    "text"
].fillna("").astype(str).str.strip()

combined = combined[
    combined["text"].str.len() > 0
]

combined = combined.drop_duplicates(
    subset=["text"]
)

combined["label"] = "unharmful"

combined.to_csv(
    OUTPUT_PATH,
    index=False
)

print("OASST conversational augmentation completed.")
print(f"Natural examples: {len(natural)}")
print(f"Augmented candidates: {len(augmented)}")
print(f"Final unique examples: {len(combined)}")
print(f"Saved to: {OUTPUT_PATH}")

print()
print("Label distribution:")
print(combined["label"].value_counts())

print()
print("First examples:")
print("=" * 70)

for i, text in enumerate(
    combined["text"].head(50),
    start=1
):
    print(f"{i}. {text}")
    