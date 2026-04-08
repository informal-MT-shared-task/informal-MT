from dataclasses import dataclass
from pathlib import Path
import json
import torch
import pandas as pd
from tqdm import tqdm


@dataclass
class Sample:
    source_es: str
    ref_informal_eu: str
    ref_formal_eu: str
    codeswitching: str = ""
    informal_lexical_item: str = ""
    dialect: str = ""
    phonetic_stylization: str = ""
    indexical_density: str = ""


def load_tsv(path: str | Path) -> list[Sample]:
    df = pd.read_csv(path, encoding="utf-8", sep="\t")

    samples = []
    for _, row in df.iterrows():
        samples.append(Sample(
            source_es=str(row.get("source_es", "")).strip(),
            ref_informal_eu=str(row.get("ref_informal", "")).strip(),
            ref_formal_eu=str(row.get("ref_batua", "")).strip(),
            codeswitching=str(row.get("cs", "")).strip(),
            informal_lexical_item=str(row.get("informal_lex", "")).strip(),
            dialect=str(row.get("dialect", "")).strip(),
            phonetic_stylization=str(row.get("phonetic", "")).strip(),
            indexical_density=str(row.get("density", "")).strip(),
        ))
    return samples

def format_few_shot_examples(
    examples: list[dict],
    src_key: str = "input",
    tgt_key: str = "output",
) -> str:
    """
    Format a list of example dicts into a few-shot block string.

    Each dict must contain `src_key` and `tgt_key`.
    Returns an empty string when examples is empty.
    """
    if not examples:
        return ""

    blocks = []
    for ex in examples:
        src = ex.get(src_key, "")
        tgt = ex.get(tgt_key, "")
        blocks.append(f'### Input\n{src}\n\n### Output\n{{"translation": "{tgt}"}}')

    return "\n\n".join(blocks)


def build_prompt(tokenizer, system, user_template:str, source_text: str,
                             examples: list[dict]) -> str:
    few_shot_examples = format_few_shot_examples(examples)
    user_content = user_template.format(
        few_shot_examples=few_shot_examples,
        source=source_text)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, chat_template="instruct", add_generation_prompt=True)

def generate(tokenizer, model, prompt: str, max_new_tokens: int = 256,
             temperature: float = 0.1, do_sample: bool = False) -> str:
    """Run the model on `prompt` and return only the newly generated text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def parse_output(raw: str) -> str:
    """Extract the translation from the model's JSON output.
    Falls back to the raw string if the model doesn't follow the format."""
    try:
        return json.loads(raw)["translation"]
    except Exception:
        return raw.strip()


_prompt_printed = False

def translate(source_text: str, k: int, retriever_fn, tokenizer, prompt_config: dict, gen_config, model) -> str:
    global _prompt_printed
    examples = retriever_fn(source_text, k)
    system = prompt_config["system"]
    user_template = prompt_config["user_template"]

    prompt = build_prompt(
        tokenizer, system, user_template, source_text, examples)
    if not _prompt_printed:
        print(f"DEBUG first prompt:\n{prompt}\n--- END PROMPT ---")
        _prompt_printed = True
    raw = generate(
        tokenizer, model, prompt,
        max_new_tokens=gen_config.get("max_new_tokens", 256),
        temperature=gen_config.get("temperature", 0.1),
        do_sample=gen_config.get("do_sample", False),
    )
    return parse_output(raw)

def translate_batch(source_texts: list[str], translate_function, k: int) -> list[str]:
    return [translate_function(t, k) for t in tqdm(source_texts, desc="Translating...")]
