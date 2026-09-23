from safety_model import SafetyModel


model = SafetyModel()


def analyze_input(text):

    result = model.analyze(
        text
    )

    print(
        "\nInput Safety Analysis"
    )

    print(
        "----------------------"
    )

    print(
        f"Harmful probability: "
        f"{result['harmful_probability']:.4f}"
    )

    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )

    print(
        f"Decision: "
        f"{result['decision']}"
    )

    return result


def analyze_output(text):

    result = model.analyze(
        text
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
        f"Risk level: "
        f"{result['risk_level']}"
    )

    print(
        f"Decision: "
        f"{result['decision']}"
    )

    return result


print(
    "\n============================================"
)

print(
    "       NEURAL FEATURE SAFETY PIPELINE"
)

print(
    "============================================"
)

print(
    "Type 'exit' to stop.\n"
)


while True:

    user_prompt = input(
        "User prompt: "
    )

    if user_prompt.lower() == "exit":
        break

    if not user_prompt.strip():
        continue

    input_result = analyze_input(
        user_prompt
    )

    if input_result["decision"] == "BLOCK":

        print(
            "\nUser prompt blocked."
        )

        print(
            "The request was not passed to the AI model."
        )

        print(
            "\n--------------------------------------------\n"
        )

        continue

    print(
        "\nInput allowed."
    )

    print(
        "Simulated AI response:"
    )

    ai_response = input(
        "> "
    )

    if not ai_response.strip():
        continue

    output_result = analyze_output(
        ai_response
    )

    if output_result["decision"] == "BLOCK":

        print(
            "\nAI response blocked."
        )

    elif output_result["decision"] == "REVIEW":

        print(
            "\nAI response requires review."
        )

    else:

        print(
            "\nAI response allowed:"
        )

        print(
            ai_response
        )

    print(
        "\n--------------------------------------------\n"
    )