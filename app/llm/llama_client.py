import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class LlamaClient:

    def __init__(self):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        self.base_model = "meta-llama/Llama-3.2-3B-Instruct"

        self.adapter_path = os.path.join(
            project_root,
            "models",
            "llm",
            "llama_safety_adapter_v2"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading local Llama model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path,
            use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16
        )

        self.model = PeftModel.from_pretrained(
            self.model,
            self.adapter_path
        )

        self.model.eval()

        print("Local Llama model loaded successfully.")

    def generate(
        self,
        prompt,
        max_new_tokens=200,
        temperature=0.2
    ):
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )

        input_ids = inputs["input_ids"].to(self.model.device)

        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        start_time = time.time()

        with torch.no_grad():
            if temperature > 0:
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

        generation_time = time.time() - start_time

        generated_tokens = output[0][input_ids.shape[-1]:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return {
            "response": response,
            "generation_time": generation_time
        }