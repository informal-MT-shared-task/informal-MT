"""
Backtranslation script for Step 1 of the multi-step pipeline.

This is run only once and the result is saved to data/augmented_dataset.tsv,
which is a copy of data/dataset.tsv but with an extra column: normalized_es.
"""

import pandas as pd
import yaml
from src.translation.llm import load_latxa
from src.translation.utils import generate, build_prompt, parse_output

with open("config/prompts.yaml") as f:
    prompts_cfg = yaml.safe_load(f)

print("Loading Latxa...")
tokenizer, model = load_latxa()

df = pd.read_csv("data/dataset.tsv", sep="\t", encoding="utf-8")

cfg = prompts_cfg["backtranslation"]
normalized_es_list = []

for i, row in df.iterrows():
    standard_eu = str(row.get("Ref-Batua-Basque", "")).strip()

    prompt = build_prompt(
        tokenizer=tokenizer,
        system=cfg["system"],
        user_template=cfg["user_template"],
        source_text=standard_eu,
        examples=[],
    )

    raw = generate(tokenizer, model, prompt, max_new_tokens=256)
    normalized_es_list.append(parse_output(raw))

    if i % 10 == 0:
        print(f"Processed {i}/{len(df)}")

df["normalized_es"] = normalized_es_list
df.to_csv("data/augmented_dataset.tsv", sep="\t", index=False, encoding="utf-8")
print("Saved data/augmented_dataset.tsv")
