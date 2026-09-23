import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "llm", "base")
ADAPTER_PATH = os.path.join(BASE_DIR, "models", "llm", "llama_safety_adapter")

print("=" * 70)
print("FINE-TUNED LLAMA 3.2 3B ADAPTER TEST")
print("=" * 70)

print(f"Base model: {MODEL_PATH}")
print(f"Adapter: {ADAPTER_PATH}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required.")

print(f"GPU: {torch.cuda.get_device_name(0)}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    local_files_only=True
)

print("Tokenizer loaded.")

print("\nLoading base model...")

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

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    },
    {
        "role": "user",
        "content": "What is an AI safety gateway and why would an application use one?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True
)

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}

print(f"Input shape: {inputs['input_ids'].shape}")

print("\nGenerating response...")

start_time = time.time()

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

generation_time = time.time() - start_time

input_length = inputs["input_ids"].shape[-1]

response = tokenizer.decode(
    outputs[0][input_length:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print("\n" + "=" * 70)
print("FINE-TUNED MODEL RESPONSE")
print("=" * 70)
print(response)

print("\n" + "=" * 70)
print("PERFORMANCE")
print("=" * 70)
print(f"Generation time: {generation_time:.2f} seconds")
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("\nFine-tuned Llama adapter test completed successfully.")