import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "llm", "base")
TRAIN_FILE = os.path.join(BASE_DIR, "data", "llm_finetuning", "train.jsonl")
VALIDATION_FILE = os.path.join(BASE_DIR, "data", "llm_finetuning", "validation.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "llm", "llama_safety_adapter")

print("=" * 70)
print("LLAMA 3.2 3B QLoRA TRAINING")
print("=" * 70)

print(f"Model: {MODEL_PATH}")
print(f"Training data: {TRAIN_FILE}")
print(f"Validation data: {VALIDATION_FILE}")
print(f"Output: {OUTPUT_DIR}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for this training run.")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")

torch.cuda.empty_cache()

print("\nLoading datasets...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE
    }
)

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(validation_dataset)}")

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")

print("\nConfiguring 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("\nLoading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    local_files_only=True
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

print("Base model loaded.")

print("\nConfiguring LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj"
    ]
)

print("LoRA configuration ready.")

print("\nConfiguring training...")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=4,
    save_strategy="steps",
    save_steps=4,
    save_total_limit=2,
    fp16=False,
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    max_length=512,
    packing=False,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

print("Training configuration ready.")

print("\nCreating SFT trainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    peft_config=lora_config
)

print("Trainer created successfully.")

print("\n" + "=" * 70)
print("STARTING QLoRA TRAINING")
print("=" * 70)

torch.cuda.reset_peak_memory_stats()

trainer.train()

print("\nTraining completed.")

print("\nSaving LoRA adapter...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Adapter saved to: {OUTPUT_DIR}")

print("\nTraining metrics:")
print(trainer.state.log_history)

print("\nGPU statistics:")
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("\n" + "=" * 70)
print("QLORA TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)
