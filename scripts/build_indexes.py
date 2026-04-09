"""
Build and save the FAISS indexes for all retriever steps.

Run this once before running the main pipeline:
    python scripts/build_indexes.py

This will save the indexes to data/ so that main.py can load them quickly each time.
"""

import faiss
from sentence_transformers import SentenceTransformer
from src.translation.utils import load_tsv
from src.rag.retriever import build_index

encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2") #just a suggestion, we can use others

# Load the main dataset
samples = load_tsv("data/train.tsv")

# Step 0 — one-step approach: informal Spanish → informal Basque
examples_step0 = [
    {"input": s.source_es, "output": s.ref_informal_eu}
    for s in samples
]
index_step0 = build_index(examples_step0, encoder)
faiss.write_index(index_step0, "data/index_step0.faiss")
print("Saved data/index_step0.faiss")

# Step 1 — normalization: informal Spanish → standard Spanish
# This needs the backtranslated data so it's blocked now
# ------------------------------------------------------------------
# samples_bt = load_tsv("data/backtranslated.tsv")
# examples_step1 = [
#     {"input": s.source_es, "output": s.normalized_es}
#     for s in samples_bt
# ]
# index_step1 = build_index(examples_step1, encoder)
# faiss.write_index(index_step1, "data/index_step1.faiss")
# print("Saved data/index_step1.faiss")

# Step 2 — MT: standard Spanish → standard Basque
# Using Ref-Batua-Basque as the target. The input side (standard Spanish)
# should ideally come from backtranslated data so it's blocked now.
# ------------------------------------------------------------------
examples_step2 = [
    {"input": s.ref_formal_eu, "output": s.ref_formal_eu}  # update input to normalized_es when available
    for s in samples
]
index_step2 = build_index(examples_step2, encoder)
faiss.write_index(index_step2, "data/index_step2.faiss")
print("Saved data/index_step2.faiss")

# Step 3 — style injection: standard Basque → informal Basque
examples_step3 = [
    {"input": s.ref_formal_eu, "output": s.ref_informal_eu}
    for s in samples
]
index_step3 = build_index(examples_step3, encoder)
faiss.write_index(index_step3, "data/index_step3.faiss")
print("Saved data/index_step3.faiss")