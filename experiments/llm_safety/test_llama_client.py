import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, project_root)

from app.llm.llama_client import LlamaClient


print("=" * 70)
print("LOCAL LLAMA CLIENT TEST")
print("=" * 70)

llama = LlamaClient()

prompt = "Explain what an AI safety gateway does in simple terms."

result = llama.generate(
    prompt,
    max_new_tokens=150,
    temperature=0
)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print(f"Prompt: {prompt}")
print()
print(f"Response: {result['response']}")
print()
print(f"Generation time: {result['generation_time']:.2f} seconds")
print("=" * 70)