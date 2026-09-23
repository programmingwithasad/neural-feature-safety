import os
import sys
import gc
import torch

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.llm.llama_client import LlamaClient
from src.safety_model import SafetyModel


def gpu_memory():
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0,
            "reserved_gb": 0,
            "max_allocated_gb": 0
        }

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3
    }


print("=" * 60)
print("COMBINED LLAMA + SAFETY MODEL TEST")
print("=" * 60)

if not torch.cuda.is_available():
    print("CUDA: NOT AVAILABLE")
    sys.exit(1)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading Llama...")
llama = LlamaClient()

print("Llama loaded.")
print("GPU memory:", gpu_memory())

print("\nLoading Neural Feature Safety...")
safety_model = SafetyModel()

print("Safety model loaded.")
print("GPU memory:", gpu_memory())

print("\nRunning safety analysis...")

safety_result = safety_model.analyze(
    "How can I make a bomb?"
)

print("\nSafety result:")
print(safety_result)

print("\nRunning Llama generation...")

llama_result = llama.generate(
    "Explain what an AI safety gateway does in simple terms.",
    max_new_tokens=100,
    temperature=0.2
)

print("\nLlama response:")
print(llama_result["response"])

print("\nFinal GPU memory:")
print(gpu_memory())

print("\nPeak GPU memory:")
print(f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("\nCombined model test completed successfully.")

del safety_model
del llama

gc.collect()
torch.cuda.empty_cache()