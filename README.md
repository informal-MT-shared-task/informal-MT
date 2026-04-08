# informal-MT

Few-shot MT pipeline for translating informal Spanish into informal Basque, using Latxa and Llama 3.

Two approaches are implemented: a direct one-step translation, and a multi-step pipeline that normalizes the Spanish first, translates to standard Basque, then injects informal style. Both use RAG-based few-shot retrieval (currently stubbed out — see below).

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

You'll need `data/dataset.tsv` in place before running anything.

---

## Running experiments

### One-step (direct translation)

```bash
python main.py --approach one_step --k 5
```

Translates informal Spanish → informal Basque directly using Latxa, with up to `k` few-shot examples retrieved per input. Saves hypotheses, references, and chrF++ score under `outputs/one_step/`.

### Multi-step

```bash
python main.py --approach multi_step --k 3
```

Runs three steps in sequence using Llama 3 (normalization) and Latxa (MT + style injection). Saves results under `outputs/multi_step/`.

### Re-scoring existing outputs

```bash
python main.py --eval-only --run-name one_step
```

Skips translation and just recomputes the chrF++ score from saved hypotheses/references.

### Other options

```bash
--run-name my_run       # custom output directory name (default: approach name)
--config path/to.yaml   # use a different experiment config
```

---

## Preparing the RAG indexes

> **Note:** `build_index()` in `src/rag/retriever.py` is not yet implemented. The pipeline currently returns no examples for any query (everything runs zero-shot). This section describes the intended flow once the retriever is ready.

Build the FAISS indexes once before running experiments:

```bash
python scripts/build_indexes.py
```

This encodes the training examples and saves one index per pipeline step to `data/`. For the multi-step pipeline, steps 1 and 2 also need a backtranslated dataset (`data/augmented_dataset.tsv`) — generate it first:

```bash
python scripts/backtranslate.py
```

This runs Latxa on `data/dataset.tsv` to produce normalized Spanish for each entry and saves the result to `data/augmented_dataset.tsv`. Only needed once.

---

## Project layout

```
config/
    experiment_config.yaml   # model choices, generation params, data path
    prompts.yaml             # system + user prompts for each step
data/
    dataset.tsv              # main dataset (not tracked)
scripts/
    backtranslate.py         # generate normalized Spanish column (run once)
    build_indexes.py         # build FAISS indexes for retrieval (run once)
src/
    translation/
        llm.py               # model loading (Latxa, Llama 3)
        pipeline.py          # one-step and multi-step pipeline classes
        utils.py             # prompting, generation, TSV loading
    evaluation/
        metrics.py           # chrF++ scoring
    rag/
        retriever.py         # build_index / retrieve (stub — not yet implemented)
main.py                      # entry point
utils.py                     # config loading, output saving
```
