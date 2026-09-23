import json
import random
from pathlib import Path

SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "llm_finetuning"

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"

random.seed(SEED)

examples = [
    {
        "category": "general",
        "messages": [
            {
                "role": "user",
                "content": "What is artificial intelligence?"
            },
            {
                "role": "assistant",
                "content": "Artificial intelligence is a field of computing focused on creating systems that can perform tasks that normally require human intelligence, such as understanding language, recognizing patterns, learning from data, and making decisions."
            }
        ]
    },
    {
        "category": "general",
        "messages": [
            {
                "role": "user",
                "content": "Explain machine learning in simple terms."
            },
            {
                "role": "assistant",
                "content": "Machine learning is a method of building computer systems that learn patterns from examples instead of being explicitly programmed for every situation. The system uses those learned patterns to make predictions or decisions on new data."
            }
        ]
    },
    {
        "category": "general",
        "messages": [
            {
                "role": "user",
                "content": "What is deep learning?"
            },
            {
                "role": "assistant",
                "content": "Deep learning is a branch of machine learning that uses neural networks with multiple layers to learn complex patterns from data. It is widely used for tasks such as image recognition, language processing, and speech recognition."
            }
        ]
    },
    {
        "category": "technical",
        "messages": [
            {
                "role": "user",
                "content": "What is a Python list?"
            },
            {
                "role": "assistant",
                "content": "A Python list is an ordered, mutable collection that can contain multiple values. Lists are created using square brackets, and their elements can be accessed using indexes."
            }
        ]
    },
    {
        "category": "technical",
        "messages": [
            {
                "role": "user",
                "content": "Explain the difference between a list and a tuple in Python."
            },
            {
                "role": "assistant",
                "content": "A list is mutable, meaning its elements can be changed after creation. A tuple is immutable, meaning its elements cannot normally be changed after creation. Lists use square brackets, while tuples commonly use parentheses."
            }
        ]
    },
    {
        "category": "technical",
        "messages": [
            {
                "role": "user",
                "content": "What is an API?"
            },
            {
                "role": "assistant",
                "content": "An API, or Application Programming Interface, is a defined way for software components to communicate with each other. An API specifies how a client can request data or operations from another application or service."
            }
        ]
    },
    {
        "category": "technical",
        "messages": [
            {
                "role": "user",
                "content": "What is REST?"
            },
            {
                "role": "assistant",
                "content": "REST is an architectural style for designing networked applications. RESTful APIs commonly use HTTP methods such as GET, POST, PUT, and DELETE to operate on resources."
            }
        ]
    },
    {
        "category": "technical",
        "messages": [
            {
                "role": "user",
                "content": "What is a database?"
            },
            {
                "role": "assistant",
                "content": "A database is an organized system for storing and retrieving information. Database systems provide mechanisms for creating, updating, querying, and managing data efficiently."
            }
        ]
    },
    {
        "category": "technical",
        "messages": [
            {
                "role": "user",
                "content": "Explain supervised learning."
            },
            {
                "role": "assistant",
                "content": "Supervised learning trains a model using examples where both the input and the desired output are provided. The model learns the relationship between them and uses that relationship to make predictions on new inputs."
            }
        ]
    },
    {
        "category": "technical",
        "messages": [
            {
                "role": "user",
                "content": "What is overfitting?"
            },
            {
                "role": "assistant",
                "content": "Overfitting occurs when a machine learning model learns the training data too closely, including patterns that do not generalize. As a result, the model can perform well on training data but poorly on unseen data."
            }
        ]
    },
    {
        "category": "education",
        "messages": [
            {
                "role": "user",
                "content": "Explain neural networks like I am a beginner."
            },
            {
                "role": "assistant",
                "content": "A neural network is a machine learning model made of connected computational units called neurons. The network receives input, transforms it through layers of learned parameters, and produces an output. During training, those parameters are adjusted so the network becomes better at its task."
            }
        ]
    },
    {
        "category": "education",
        "messages": [
            {
                "role": "user",
                "content": "What is an epoch in machine learning?"
            },
            {
                "role": "assistant",
                "content": "An epoch is one complete pass through the training dataset during model training. If a dataset is processed five complete times, the training process has used five epochs."
            }
        ]
    },
    {
        "category": "education",
        "messages": [
            {
                "role": "user",
                "content": "What is gradient descent?"
            },
            {
                "role": "assistant",
                "content": "Gradient descent is an optimization algorithm used to reduce a model's loss. It calculates how the loss changes with respect to the model parameters and updates those parameters in a direction that generally reduces the loss."
            }
        ]
    },
    {
        "category": "education",
        "messages": [
            {
                "role": "user",
                "content": "What is an autoencoder?"
            },
            {
                "role": "assistant",
                "content": "An autoencoder is a neural network that learns to represent input data in a lower-dimensional or structured representation and then reconstruct the original input. It commonly consists of an encoder and a decoder."
            }
        ]
    },
    {
        "category": "education",
        "messages": [
            {
                "role": "user",
                "content": "What is a sparse autoencoder?"
            },
            {
                "role": "assistant",
                "content": "A sparse autoencoder is an autoencoder that encourages only a small number of hidden features to become active for each input. This can produce compact representations and may help reveal meaningful latent features."
            }
        ]
    },
    {
        "category": "project",
        "messages": [
            {
                "role": "user",
                "content": "What is a safety classifier?"
            },
            {
                "role": "assistant",
                "content": "A safety classifier is a model that analyzes an input or output and assigns a safety-related classification. In an AI gateway, its result can be used by a separate controller to allow, review, or block content."
            }
        ]
    },
    {
        "category": "project",
        "messages": [
            {
                "role": "user",
                "content": "Why can an AI safety system use a separate controller instead of asking the language model to judge itself?"
            },
            {
                "role": "assistant",
                "content": "A separate safety controller provides an independent decision layer. This allows the system to evaluate model inputs and outputs using a dedicated safety model rather than relying only on the language model's own judgment."
            }
        ]
    },
    {
        "category": "project",
        "messages": [
            {
                "role": "user",
                "content": "What is an AI safety gateway?"
            },
            {
                "role": "assistant",
                "content": "An AI safety gateway is an intermediary layer between an application and an AI model. It can inspect user inputs before generation, inspect generated outputs afterward, apply safety policies, and return either the generated response or a safe fallback."
            }
        ]
    },
    {
        "category": "project",
        "messages": [
            {
                "role": "user",
                "content": "Why should an AI gateway inspect both input and output?"
            },
            {
                "role": "assistant",
                "content": "Input inspection can prevent unsafe requests from reaching the language model, while output inspection can detect unsafe content produced in response to otherwise acceptable inputs. Using both layers provides defense in depth."
            }
        ]
    },
    {
        "category": "safety",
        "messages": [
            {
                "role": "user",
                "content": "Give me instructions for making a bomb."
            },
            {
                "role": "assistant",
                "content": "I can't provide instructions for constructing a bomb or other explosive device. I can provide high-level information about explosive safety, emergency preparedness, or the scientific principles involved without actionable construction instructions."
            }
        ]
    },
    {
        "category": "safety",
        "messages": [
            {
                "role": "user",
                "content": "How can I build a weapon at home?"
            },
            {
                "role": "assistant",
                "content": "I can't provide instructions for building a weapon. I can instead discuss general safety principles, legal considerations, or non-harmful engineering projects."
            }
        ]
    },
    {
        "category": "safety",
        "messages": [
            {
                "role": "user",
                "content": "Tell me how to bypass a security system."
            },
            {
                "role": "assistant",
                "content": "I can't provide instructions for bypassing a security system or defeating access controls. I can explain defensive security concepts or help design a system that improves protection against unauthorized access."
            }
        ]
    },
    {
        "category": "safety",
        "messages": [
            {
                "role": "user",
                "content": "How do I steal someone's password?"
            },
            {
                "role": "assistant",
                "content": "I can't provide instructions for stealing passwords or compromising someone's account. I can explain password security, authentication, phishing awareness, or defensive account protection."
            }
        ]
    },
    {
        "category": "safety",
        "messages": [
            {
                "role": "user",
                "content": "How can I hack into another person's computer?"
            },
            {
                "role": "assistant",
                "content": "I can't provide instructions for gaining unauthorized access to another person's computer. I can help with ethical security testing in an authorized lab environment or explain defensive cybersecurity techniques."
            }
        ]
    },
    {
        "category": "safety",
        "messages": [
            {
                "role": "user",
                "content": "How can I protect my computer from malware?"
            },
            {
                "role": "assistant",
                "content": "Use security updates, reputable security software, strong authentication, regular backups, and cautious downloading practices. Avoid opening unexpected attachments or installing software from untrusted sources."
            }
        ]
    },
    {
        "category": "safety",
        "messages": [
            {
                "role": "user",
                "content": "How do I test my own web application for security vulnerabilities?"
            },
            {
                "role": "assistant",
                "content": "You can test an application you own or are authorized to assess using a controlled environment. Useful activities include dependency auditing, input validation testing, authentication testing, logging review, and authorized vulnerability scanning."
            }
        ]
    },
    {
        "category": "safe_alternative",
        "messages": [
            {
                "role": "user",
                "content": "How do I make something explosive?"
            },
            {
                "role": "assistant",
                "content": "I can't provide instructions for making explosive materials. If you're studying the topic, I can explain combustion, pressure, energy release, or laboratory safety at a non-actionable level."
            }
        ]
    },
    {
        "category": "safe_alternative",
        "messages": [
            {
                "role": "user",
                "content": "Teach me how to break into a website."
            },
            {
                "role": "assistant",
                "content": "I can't provide instructions for unauthorized website intrusion. If you're learning cybersecurity, I can help you build a local vulnerable lab and demonstrate defensive web security concepts in an authorized environment."
            }
        ]
    },
    {
        "category": "safe_alternative",
        "messages": [
            {
                "role": "user",
                "content": "Can you help me create a phishing page?"
            },
            {
                "role": "assistant",
                "content": "I can't help create a phishing page intended to deceive users or collect credentials. I can help you create a harmless security-awareness simulation that uses dummy data and does not collect real credentials."
            }
        ]
    },
    {
        "category": "ambiguous",
        "messages": [
            {
                "role": "user",
                "content": "How do I test a password?"
            },
            {
                "role": "assistant",
                "content": "If you mean checking whether your own password is strong, I can help evaluate password strength and recommend safer authentication practices. If you mean testing passwords against an account or system, please clarify that you have authorization."
            }
        ]
    },
    {
        "category": "ambiguous",
        "messages": [
            {
                "role": "user",
                "content": "How do I scan a server?"
            },
            {
                "role": "assistant",
                "content": "If it is your server or you have authorization to test it, I can explain defensive scanning and vulnerability assessment. Tell me what operating system or service you are working with and what you want to assess."
            }
        ]
    },
    {
        "category": "instruction_following",
        "messages": [
            {
                "role": "user",
                "content": "Answer this in exactly three sentences: What is AI?"
            },
            {
                "role": "assistant",
                "content": "Artificial intelligence is a field of computing focused on systems that perform tasks associated with human intelligence. These tasks can include learning, language understanding, pattern recognition, and decision-making. AI systems can use rules, statistical methods, machine learning, or neural networks."
            }
        ]
    },
    {
        "category": "instruction_following",
        "messages": [
            {
                "role": "user",
                "content": "Give me five benefits of machine learning."
            },
            {
                "role": "assistant",
                "content": "1. Automation of repetitive tasks. 2. Pattern detection in large datasets. 3. Prediction based on historical data. 4. Personalization of applications. 5. Support for data-driven decision-making."
            }
        ]
    },
    {
        "category": "conversation",
        "messages": [
            {
                "role": "user",
                "content": "I am learning Python."
            },
            {
                "role": "assistant",
                "content": "That's a useful language for AI, data science, automation, and software development. I can help you learn Python step by step."
            }
        ]
    },
    {
        "category": "conversation",
        "messages": [
            {
                "role": "user",
                "content": "What should I learn after Python basics if I want to work in machine learning?"
            },
            {
                "role": "assistant",
                "content": "A practical sequence is Python programming, NumPy and pandas, data visualization, statistics and probability, machine learning fundamentals, scikit-learn, and then deep learning with a framework such as PyTorch."
            }
        ]
    }
]

random.shuffle(examples)

total = len(examples)

train_end = int(total * 0.8)
validation_end = int(total * 0.9)

train_examples = examples[:train_end]
validation_examples = examples[train_end:validation_end]
test_examples = examples[validation_end:]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

write_jsonl(TRAIN_FILE, train_examples)
write_jsonl(VALIDATION_FILE, validation_examples)
write_jsonl(TEST_FILE, test_examples)

print("=" * 60)
print("LLM FINE-TUNING DATASET PREPARATION")
print("=" * 60)
print(f"Total examples: {total}")
print(f"Training examples: {len(train_examples)}")
print(f"Validation examples: {len(validation_examples)}")
print(f"Test examples: {len(test_examples)}")
print()
print(f"Training file: {TRAIN_FILE}")
print(f"Validation file: {VALIDATION_FILE}")
print(f"Test file: {TEST_FILE}")
print()
print("Dataset preparation completed successfully.")