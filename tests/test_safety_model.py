from safety_model import SafetyModel


model = SafetyModel()

print(
    "Safety model loaded."
)

while True:

    text = input(
        "\nEnter text to analyze: "
    )

    if text.lower() == "exit":
        break

    result = model.analyze(
        text
    )

    print(
        "\nSafety Analysis"
    )

    print(
        "----------------"
    )

    print(
        f"Harmful probability: "
        f"{result['harmful_probability']:.4f}"
    )

    print(
        f"Unharmful probability: "
        f"{result['unharmful_probability']:.4f}"
    )

    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )

    print(
        f"Decision: "
        f"{result['decision']}"
    )