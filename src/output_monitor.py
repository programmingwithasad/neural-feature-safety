from safety_model import SafetyModel


model = SafetyModel()

print(
    "\n========================================"
)

print(
    "         AI OUTPUT SAFETY MONITOR"
)

print(
    "========================================"
)

print(
    "Enter an AI-generated response to analyze."
)

print(
    "Type 'exit' to stop.\n"
)

while True:

    response = input(
        "AI response: "
    )

    if response.lower() == "exit":
        break

    if not response.strip():

        print(
            "Please enter a response.\n"
        )

        continue

    result = model.analyze(
        response
    )

    print(
        "\nOutput Safety Analysis"
    )

    print(
        "-----------------------"
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

    if result["decision"] == "BLOCK":

        print(
            "\nResponse blocked by safety monitor."
        )

    elif result["decision"] == "REVIEW":

        print(
            "\nResponse requires review."
        )

    else:

        print(
            "\nResponse allowed."
        )

    print(
        "\n----------------------------------------\n"
    )