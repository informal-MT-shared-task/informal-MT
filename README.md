# informal-MT

Few-shot MT pipeline for translating informal Spanish into informal Basque, using Latxa 7B and Llama 3 8B.

Two approaches are implemented: a direct **one-step** translation, and a **multi-step** pipeline that normalizes the Spanish first, translates to standard Basque, then injects informal style. Both support pluggable few-shot retrieval strategies.

---

## Setup

Install PyTorch first, since the right variant depends on your hardware:

```bash
# Mac
pip install torch>=2.0.0

# Linux, CUDA 12.1
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# Colab / Linux CPU
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install the rest:

```bash
pip install -r requirements.txt
```

On Linux or Colab (CUDA), also install bitsandbytes for 4-bit quantization:

```bash
pip install bitsandbytes>=0.43.0
```

You'll need `data/train.tsv` and `data/test.tsv` in place before running anything.

---

## Running experiments

### One-step (direct translation)

```bash
python main.py --approach one_step --k 15
```

Translates informal Spanish → informal Basque directly using Latxa 7B, with `k` few-shot examples per input. Saves hypotheses, references, and chrF++ score under `outputs/`.

### Multi-step

```bash
python main.py --approach multi_step --k 5
```

Runs three steps in sequence: Llama 3 8B (normalization) → Latxa 7B (formal MT) → Latxa 7B (style injection).

### Re-scoring existing outputs

```bash
python main.py --eval-only --k 15
```

Skips translation and recomputes the chrF++ score from saved hypotheses and references. Automatically infers the approach from the filename in `outputs/`.

### Other options

```bash
--retrieval-strategy {hardcoded,random,phenomena,faiss}   # override config value
--config path/to/experiment_config.yaml                   # use a different config
```

---

## Retrieval strategies

Controlled by `--retrieval-strategy` (or `rag.strategy` in `experiment_config.yaml`):

| Strategy | Description |
|---|---|
| `hardcoded` | 20 hand-curated examples from train.tsv, sliced to k. Default. |
| `random` | Random sample from the training pool each run. |
| `phenomena` | PhenomenaRetriever: fixed anchor examples + dynamic slots based on detected informality phenomena (elongation, informal lexicon). |
| `faiss` | Semantic nearest-neighbour search via FAISS index (requires pre-built index). |

### Configuring the PhenomenaRetriever

In `config/experiment_config.yaml`, fill in `hardcoded_example_indices` with at least 3 training example indices (0-based row numbers from `train.tsv`). More indices = better fallback coverage when no phenomena are detected:

```yaml
rag:
  strategy: "hardcoded"
  n_examples: 12          # increase up to 20 for larger models
  hardcoded_example_indices: [78, 87, 56, 3, 1, 20, 101, 116, 52, 17, 90, 66, 6, 9, 27, 29, 111, 81, 61, 26]
```

### Building FAISS indexes (faiss strategy only)

```bash
python scripts/build_indexes.py
```

Encodes training examples and saves one index per pipeline step to `data/`. Only needed once, and only if using `--retrieval-strategy faiss`.

---

## Preparing data

### Converting shared task files to TSV

If you receive source and reference files as plain text (one sentence per line):

```bash
python scripts/txt_to_tsv.py --src source.txt --ref reference.txt --out data/test.tsv
```

### Generating augmented data for multi-step (run once)

```bash
python scripts/backtranslate.py
```

Runs Latxa on `data/train.tsv` to produce normalized Spanish for each entry and saves the result to `data/augmented_dataset.tsv`. Required for the multi-step pipeline.

---

## Project layout

```
config/
    experiment_config.yaml      # model choices, generation params, data paths, RAG config
    prompts.yaml                # system + user prompts for each pipeline step
data/
    train.tsv                   # training data with annotations (not tracked)
    test.tsv                    # test data (not tracked)
scripts/
    txt_to_tsv.py               # convert plain-text source+reference to TSV
    backtranslate.py            # generate normalized Spanish column (multi-step only)
    build_indexes.py            # build FAISS indexes for faiss retrieval strategy
src/
    translation/
        llm.py                  # model loading with 4-bit quantization (Latxa, Llama 3)
        pipeline.py             # one-step and multi-step pipeline classes
        utils.py                # prompting, generation, parsing, post-processing
    evaluation/
        metrics.py              # chrF++ scoring via sacrebleu
    rag/
        retriever.py            # FAISS-based semantic retrieval
        engine.py               # PhenomenaRetriever (phenomena-aware hybrid retrieval)
        phenomena.py            # detect_phenomena() and lexical_overlap() helpers
main.py                         # entry point and CLI
utils.py                        # config loading, output saving
```
