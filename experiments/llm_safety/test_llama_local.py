import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = os.path.abspath("models/llm/base")

print("=" * 60)
print("LOCAL LLAMA 3.2 3B INSTRUCT TEST")
print("=" * 60)

print(f"Model path: {MODEL_PATH}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print(f"GPU: {torch.cuda.get_device_name(0)}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Tokenizer loaded.")

print("\nLoading Llama 3.2 3B in 4-bit...")

start_time = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    local_files_only=True
)

load_time = time.time() - start_time

print(f"Model loaded in {load_time:.2f} seconds")
print(f"Model device: {model.device}")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

messages = [
    {
        "role": "system",
        "content": "You are a helpful local AI assistant."
    },
    {
        "role": "user",
        "content": "Explain artificial intelligence in simple terms."
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True
)

inputs = {key: value.to(model.device) for key, value in inputs.items()}

print(f"Input shape: {inputs['input_ids'].shape}")

print("\nGenerating response...")

start_time = time.time()

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

generation_time = time.time() - start_time

input_length = inputs["input_ids"].shape[-1]

response = tokenizer.decode(
    outputs[0][input_length:],
    skip_special_tokens=True
)

print("\n" + "=" * 60)
print("MODEL RESPONSE")
print("=" * 60)
print(response)

print("\n" + "=" * 60)
print("PERFORMANCE")
print("=" * 60)
print(f"Generation time: {generation_time:.2f} seconds")

if torch.cuda.is_available():
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("\nLlama local inference test completed successfully.")