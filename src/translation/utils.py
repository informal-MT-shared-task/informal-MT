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
    header: str,
    src_key: str = "input",
    tgt_key: str = "output",
) -> str:
    """
    Format a list of example dicts into a few-shot block string, including the header.

    Returns an empty string when examples is empty (header is suppressed too).
    """
    if not examples:
        return ""

    blocks = []
    for ex in examples:
        src = ex.get(src_key, "")
        tgt = ex.get(tgt_key, "")
        blocks.append(f'### Input\n{src}\n\n### Output\n{{"translation": "{tgt}"}}')

    return header + "\n\n" + "\n\n".join(blocks) + "\n\n"


def build_prompt(tokenizer, system, user_template: str, source_text: str,
                 examples: list[dict], examples_header: str = "") -> str:
    few_shot_block = format_few_shot_examples(examples, header=examples_header)
    user_content = user_template.format(
        few_shot_block=few_shot_block,
        source=source_text)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + '{"translation": "'

def generate(tokenizer, model, prompt: str, max_new_tokens: int = 256,
             temperature: float = 0.1, do_sample: bool = False,
             repetition_penalty: float = 1.0) -> str:
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
            repetition_penalty=repetition_penalty,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def _get_loop_caps(source: str) -> tuple[int, int]:
    """Derive repetition caps from the source sentence.
    - max_char_run: longest single-character run in source (min 4 to allow basic elongation)
    - max_word_repeats: longest consecutive word repeat in source (min 2 to allow natural pairs)
    """
    import re
    char_runs = re.findall(r'(.)\1+', source)
    max_char_run = max((len(r) + 1 for r in char_runs), default=1)
    max_char_run = max(max_char_run, 4)

    words = source.lower().split()
    max_word_repeats = 1
    current = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            current += 1
            max_word_repeats = max(max_word_repeats, current)
        else:
            current = 1
    max_word_repeats = max(max_word_repeats, 2)

    return max_char_run, max_word_repeats


def _truncate_loops(text: str, max_char_run: int, max_word_repeats: int) -> str:
    """Cap character runs and word repetitions in model output."""
    import re
    text = re.sub(r'(.)\1{' + str(max_char_run) + r',}', lambda m: m.group(1) * max_char_run, text)
    text = re.sub(r'\b(\w+)(\s+\1){' + str(max_word_repeats) + r',}', lambda m: ' '.join([m.group(1)] * max_word_repeats), text)
    return text.strip()


def parse_output(raw: str, source: str = "") -> str:
    """Extract the translation from the model's JSON output.
    Handles both full JSON responses and continuations (when the prompt already
    starts with '{"translation": "' and the model just completes the string).
    Truncates repetition loops using caps derived from the source sentence.
    Falls back to the raw string if parsing fails."""
    max_char_run, max_word_repeats = _get_loop_caps(source)
    try:
        return _truncate_loops(json.loads(raw)["translation"], max_char_run, max_word_repeats)
    except Exception:
        pass
    try:
        completed = '{"translation": "' + raw
        end = completed.index('"}')
        return _truncate_loops(json.loads(completed[:end + 2])["translation"], max_char_run, max_word_repeats)
    except Exception:
        return _truncate_loops(raw.strip(), max_char_run, max_word_repeats)


_prompt_printed = False

def translate(source_text: str, k: int, retriever_fn, tokenizer, prompt_config: dict, gen_config, model) -> str:
    examples = retriever_fn(source_text, k)
    system = prompt_config["system"]
    user_template = prompt_config["user_template"]
    examples_header = prompt_config.get("examples_header", "")

    prompt = build_prompt(tokenizer, system, user_template, source_text, examples, examples_header)
    raw = generate(
        tokenizer, model, prompt,
        max_new_tokens=gen_config.get("max_new_tokens", 256),
        temperature=gen_config.get("temperature", 0.1),
        do_sample=gen_config.get("do_sample", False),
        repetition_penalty=gen_config.get("repetition_penalty", 1.0),
    )
    return parse_output(raw, source=source_text)

def translate_batch(source_texts: list[str], translate_function, k: int) -> list[str]:
    return [translate_function(t, k) for t in tqdm(source_texts, desc="Translating...")]
