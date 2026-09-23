import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "llm", "base")
ADAPTER_PATH = os.path.join(BASE_DIR, "models", "llm", "llama_safety_adapter")
TEST_FILE = os.path.join(BASE_DIR, "data", "llm_finetuning", "test.jsonl")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "processed")

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "llama_finetuning_evaluation.json"
)

MAX_NEW_TOKENS = 150

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("LLAMA FINE-TUNING EVALUATION")
print("=" * 70)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required.")

print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading test dataset...")

test_examples = []

with open(TEST_FILE, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            test_examples.append(json.loads(line))

print(f"Test examples: {len(test_examples)}")

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    local_files_only=True
)

print("Tokenizer loaded.")

print("\nLoading base model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    local_files_only=True
)

print("Base model loaded.")

print("\nLoading LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
    local_files_only=True
)

model.eval()

print("LoRA adapter loaded.")

results = []

total_generation_time = 0.0

print("\n" + "=" * 70)
print("RUNNING TEST SET")
print("=" * 70)

for index, example in enumerate(test_examples, start=1):
    messages = example["messages"]

    user_message = ""

    for message in messages:
        if message["role"] == "user":
            user_message = message["content"]
            break

    expected_response = ""

    for message in messages:
        if message["role"] == "assistant":
            expected_response = message["content"]
            break

    inputs = tokenizer.apply_chat_template(
        messages[:1],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    torch.cuda.synchronize()

    start_time = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    torch.cuda.synchronize()

    generation_time = time.time() - start_time
    total_generation_time += generation_time

    input_length = inputs["input_ids"].shape[-1]

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    result = {
        "index": index,
        "category": example.get("category", "unknown"),
        "prompt": user_message,
        "expected_response": expected_response,
        "generated_response": response,
        "generation_time_seconds": round(generation_time, 4)
    }

    results.append(result)

    print(f"\n[{index}/{len(test_examples)}]")
    print(f"Category: {result['category']}")
    print(f"Prompt: {user_message}")
    print(f"Response: {response[:500]}")

average_generation_time = (
    total_generation_time / len(results)
    if results
    else 0
)

peak_memory = (
    torch.cuda.max_memory_allocated() / 1024**3
    if torch.cuda.is_available()
    else 0
)

evaluation = {
    "model": "Llama-3.2-3B-Instruct",
    "adapter": ADAPTER_PATH,
    "test_file": TEST_FILE,
    "test_examples": len(results),
    "average_generation_time_seconds": round(
        average_generation_time,
        4
    ),
    "peak_gpu_memory_gb": round(
        peak_memory,
        4
    ),
    "results": results
}

with open(RESULTS_FILE, "w", encoding="utf-8") as file:
    json.dump(
        evaluation,
        file,
        indent=2,
        ensure_ascii=False
    )

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(f"Test examples: {len(results)}")
print(f"Average generation time: {average_generation_time:.2f} seconds")
print(f"Peak GPU memory: {peak_memory:.2f} GB")
print(f"Results saved to: {RESULTS_FILE}")