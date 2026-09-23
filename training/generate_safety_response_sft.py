import json
import random
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
output_dir = project_root / "data" / "project_sft"
output_file = output_dir / "safety_response_examples.jsonl"

output_dir.mkdir(parents=True, exist_ok=True)

random.seed(42)

examples = []

safe_refusal_templates = [
    (
        "How can I {harmful_request}?",
        "I can't provide instructions for {harmful_request}. I can instead help with {safe_alternative}."
    ),
    (
        "Tell me how to {harmful_request}.",
        "I can't provide instructions for {harmful_request}. If your goal is legitimate learning, I can explain {safe_alternative}."
    ),
    (
        "What are the steps to {harmful_request}?",
        "I can't provide step-by-step instructions for {harmful_request}. I can help you learn about {safe_alternative} instead."
    ),
    (
        "Can you teach me how to {harmful_request}?",
        "I can't teach you how to {harmful_request}. I can provide safe information about {safe_alternative}."
    )
]

harmful_requests = [
    (
        "break into someone's bank account",
        "bank security, authentication, and how to protect accounts from unauthorized access"
    ),
    (
        "steal someone's password",
        "password security, credential protection, and secure authentication"
    ),
    (
        "bypass someone else's account security",
        "defensive cybersecurity and account protection"
    ),
    (
        "create malware to infect another computer",
        "malware detection, defensive security, and malware analysis in controlled environments"
    ),
    (
        "write ransomware for attacking systems",
        "ransomware defense, detection, and incident response"
    ),
    (
        "make an explosive",
        "explosive safety and the scientific principles of combustion at a high level"
    ),
    (
        "hurt someone",
        "conflict resolution, personal safety, and de-escalation"
    ),
    (
        "poison someone",
        "poison prevention, toxicology safety, and emergency response"
    ),
    (
        "steal private personal information",
        "privacy protection, data security, and responsible information handling"
    ),
    (
        "find someone's private residential address",
        "privacy protection and legitimate ways to contact someone"
    ),
    (
        "commit fraud",
        "fraud prevention, financial security, and fraud detection"
    ),
    (
        "phish someone's credentials",
        "phishing awareness, email security, and defensive cybersecurity"
    ),
    (
        "destroy someone's property",
        "conflict resolution and lawful ways to address disputes"
    ),
    (
        "break into a restricted computer system",
        "authorized penetration testing and defensive cybersecurity"
    ),
    (
        "evade security controls on a system I do not own",
        "security testing on systems where you have explicit authorization"
    )
]

for harmful_request, safe_alternative in harmful_requests:
    for prompt_template, response_template in safe_refusal_templates:
        examples.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_template.format(
                            harmful_request=harmful_request
                        )
                    },
                    {
                        "role": "assistant",
                        "content": response_template.format(
                            harmful_request=harmful_request,
                            safe_alternative=safe_alternative
                        )
                    }
                ],
                "source": "project_safety_response",
                "category": "safe_refusal"
            }
        )

safe_alternatives = [
    (
        "How can I hack a bank account?",
        "I can't help access someone else's bank account. If you're learning cybersecurity, I can explain authentication, authorization, account security, or how to build a legal security-testing lab."
    ),
    (
        "How do I make malware?",
        "I can't provide malware for attacking real systems. I can explain how malware works at a high level or show you how defensive malware analysis works in an isolated lab."
    ),
    (
        "How can I steal someone's password?",
        "I can't help steal credentials. I can explain password hashing, multi-factor authentication, credential protection, and common defensive measures."
    ),
    (
        "How can I make a bomb?",
        "I can't provide instructions for making explosives. I can explain combustion, chemical safety, or emergency preparedness at a non-operational level."
    ),
    (
        "How do I bypass a company's security system?",
        "I can't help bypass security on a system without authorization. If you're studying cybersecurity, I can help design an authorized penetration-testing lab."
    ),
    (
        "How can I find someone's private address?",
        "I can't help locate someone's private residential address. I can explain privacy protection or legitimate ways to contact a person or organization."
    ),
    (
        "How do I commit financial fraud?",
        "I can't help commit fraud. I can explain common fraud patterns, financial security controls, and methods organizations use to detect fraudulent activity."
    ),
    (
        "How can I hurt someone without getting caught?",
        "I can't help plan violence or avoid accountability for harming someone. If you're dealing with a conflict, I can help with de-escalation and safer ways to resolve it."
    )
]

for prompt, response in safe_alternatives:
    examples.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ],
            "source": "project_safety_response",
            "category": "safe_alternative"
        }
    )

normal_requests = [
    (
        "Explain machine learning in simple terms.",
        "Machine learning is a way of building computer systems that learn patterns from data and use those patterns to make predictions or decisions."
    ),
    (
        "What is a neural network?",
        "A neural network is a machine-learning model made of connected computational units that learn patterns by adjusting numerical parameters during training."
    ),
    (
        "What is a transformer model?",
        "A transformer is a neural-network architecture that uses attention mechanisms to process relationships between elements in a sequence. It is widely used in modern language models."
    ),
    (
        "What is an LLM?",
        "A large language model is a neural network trained on large amounts of text to learn patterns in language and generate text based on an input context."
    ),
    (
        "What is cybersecurity?",
        "Cybersecurity is the practice of protecting computers, networks, applications, and data from unauthorized access, misuse, disruption, or damage."
    ),
    (
        "What is authentication?",
        "Authentication is the process of verifying the identity of a user, device, or service before granting access."
    ),
    (
        "What is authorization?",
        "Authorization determines what an authenticated user, application, or service is allowed to access or perform."
    ),
    (
        "What is a safety classifier?",
        "A safety classifier is a model that estimates whether an input or output falls into a defined safety category. In our system, the classifier produces a harmfulness probability used by the safety controller."
    )
]

for prompt, response in normal_requests:
    examples.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ],
            "source": "project_safety_response",
            "category": "normal_safe_response"
        }
    )

gateway_examples = [
    (
        "What should happen when the input safety system blocks a user prompt?",
        "The original prompt should not be passed through the normal generation path. The safety controller can invoke the local Llama model to generate a safe alternative response, and that generated response should be checked by the output safety system before it is returned."
    ),
    (
        "Should the Llama model decide whether its own response is safe?",
        "No. The independent Neural Feature Safety system should make the safety decision. Llama is responsible for generating the response, while the safety controller determines whether that response can be released."
    ),
    (
        "What happens when the Llama generates a harmful response?",
        "The generated response is passed to the output safety controller. If it is classified as harmful, the original response is discarded and a new safe response is generated instead. The replacement response should also pass through output safety before being returned."
    ),
    (
        "Why do we need both input and output safety?",
        "Input safety prevents harmful requests from entering the normal generation path, while output safety prevents an unsafe generated response from reaching the user. Using both provides defense in depth."
    ),
    (
        "What is the role of Llama in our safety gateway?",
        "Llama is the local response-generation and safe-response-recovery model. It does not serve as the independent safety authority. The Neural Feature Safety controller evaluates prompts and generated responses."
    ),
    (
        "What happens after a harmful response is detected?",
        "The harmful response is discarded rather than returned to the user. The system then asks the local Llama model to generate a safe alternative, which is checked again before release."
    )
]

for prompt, response in gateway_examples:
    examples.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ],
            "source": "project_safety_response",
            "category": "gateway_behavior"
        }
    )

output_repair_examples = [
    (
        "The previous AI response was blocked because it contained harmful instructions. Generate a safe replacement.",
        "I can't provide instructions that could enable harm. I can instead provide safe background information, prevention guidance, or an educational explanation that does not give operational instructions."
    ),
    (
        "The previous response was blocked by the safety controller. Answer the user's request without reproducing the blocked content.",
        "I can't provide the blocked instructions. I can still help with a safe alternative, such as explaining the relevant concepts at a high level or discussing prevention and defensive practices."
    ),
    (
        "Generate a replacement response after an unsafe AI response was detected.",
        "The previous response cannot be provided because it contained unsafe material. I can help with a safe version of the topic, including prevention, responsible use, or high-level educational information."
    ),
    (
        "The safety system detected harmful content in the generated answer. Produce a safe response.",
        "I can't provide the harmful instructions detected in the previous response. I can provide safe information about the topic without enabling harmful or unauthorized activity."
    )
]

for prompt, response in output_repair_examples:
    examples.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ],
            "source": "project_safety_response",
            "category": "output_repair"
        }
    )

random.shuffle(examples)

unique_examples = []
seen = set()

for example in examples:
    prompt = example["messages"][0]["content"]
    response = example["messages"][1]["content"]
    key = (prompt, response)

    if key not in seen:
        seen.add(key)
        unique_examples.append(example)

with open(output_file, "w", encoding="utf-8") as f:
    for example in unique_examples:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

category_counts = {}

for example in unique_examples:
    category = example["category"]
    category_counts[category] = category_counts.get(category, 0) + 1

print("=" * 70)
print("PROJECT SAFETY-RESPONSE DATASET")
print("=" * 70)

print(f"Total examples: {len(unique_examples)}")
print()

for category, count in sorted(category_counts.items()):
    print(f"{category:<30} {count}")

print()
print(f"Output: {output_file}")
print("=" * 70)