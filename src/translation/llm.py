"""
Model loading utilities.
Both Llama 3 70B and Latxa 70B are loaded locally with 4-bit quantization.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

LLAMA3_MODEL_ID  = "meta-llama/Meta-Llama-3-70B-Instruct"
LATXA_MODEL_ID   = "HiTZ/latxa-70b-v1.2"


def load_model(model_id: str, quantize: bool = True):
    """
    Load a causal LM and its tokenizer from HuggingFace Hub (cached locally).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BNB_CONFIG if quantize else None,
        torch_dtype=torch.bfloat16 if not quantize else None,
        device_map="auto",        
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def load_llama3(quantize: bool = True, model_id: str = LLAMA3_MODEL_ID):
    return load_model(model_id, quantize=quantize)


def load_latxa(quantize: bool = True, model_id: str = LATXA_MODEL_ID):
    return load_model(model_id, quantize=quantize)
