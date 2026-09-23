import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

base_model = "meta-llama/Llama-3.2-3B-Instruct"

train_file = os.path.join(
    project_root,
    "data",
    "llm_finetuning_final",
    "train.jsonl"
)

validation_file = os.path.join(
    project_root,
    "data",
    "llm_finetuning_final",
    "validation.jsonl"
)

output_dir = os.path.join(
    project_root,
    "models",
    "llm",
    "llama_safety_adapter_v2"
)

os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("LLAMA 3.2 3B SAFETY RESPONSE FINE-TUNING V2")
print("=" * 70)

print(f"Base model:       {base_model}")
print(f"Training dataset: {train_file}")
print(f"Validation data:  {validation_file}")
print(f"Output adapter:   {output_dir}")
print()

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")

print()
print("Loading 4-bit quantized model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16
)

model.config.use_cache = False

print("Model loaded.")
print(f"Model device: {model.device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
    print(
        f"Initial GPU memory: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

print()
print("Loading final SFT dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": train_file,
        "validation": validation_file
    }
)

print(f"Training examples:   {len(dataset['train']):,}")
print(f"Validation examples: {len(dataset['validation']):,}")

print()
print("Configuring LoRA...")

peft_config = LoraConfig(
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

print()
print("Configuring SFT trainer...")

training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    max_length=512,
    packing=False,
    report_to="none",
    dataset_text_field=None
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=peft_config
)

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

train_result = trainer.train()

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Training loss: {train_result.training_loss:.4f}")
print(
    f"Training runtime: "
    f"{train_result.metrics.get('train_runtime', 0):.2f} seconds"
)

if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory: {peak_memory:.2f} GB")

print()
print("Running final validation evaluation...")

eval_result = trainer.evaluate()

print(f"Validation loss: {eval_result.get('eval_loss', 0):.4f}")

print()
print("Saving LoRA adapter...")

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print()
print("=" * 70)
print("ADAPTER SAVED")
print("=" * 70)
print(output_dir)
print("=" * 70)