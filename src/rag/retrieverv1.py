"""
The idea is to have two functions:
    - build_index(): encodes all training examples and creates a FAISS index to store them
    - retrieve(): given a query, finds the k most similar examples in the index (k comes from main.py, inserted by user)

These are then used in main.py to build a retriever function for each pipeline step.

OUTPUT
The rest of the codebase expects retrieve() to return a list of dicts like this:
    [
        {"input": "q xtal q vienes!!", "output": "ze ondo etortzen zarela!!"},
        {"input": "no puedo ir hoy",   "output": "ezin dut joan gaur"},
        ...
    ]

WHICH EXAMPLES TO USE FOR EACH STEP
We need four retrievers, one per pipeline step. Each one searches a different example pool:

    retriever_fn_step0 — one-step approach:
        "input":  informal Spanish     (column: source-ISMD-Spanish)
        "output": informal Basque      (column: Ref-ISMD-Basque(ORIGINAL))

    retriever_fn_step1 — multi-step, step 1 (normalization):
        "input":  informal Spanish     (column: source-ISMD-Spanish)
        "output": standard Spanish     (column: normalized_es in dataset_augmented.tsv)
        note: this one needs the backtranslation data that we don't have yet

    retriever_fn_step2 — multi-step, step 2 (MT):
        "input":  standard Spanish     (column: normalized_es in dataset_augmented.tsv)
        "output": standard Basque      (column: Ref-Batua-Basque)
        note: this one needs the backtranslation data that we don't have yet

    retriever_fn_step3 — multi-step, step 3 (style injection):
        "input":  standard Basque      (column: Ref-Batua-Basque)
        "output": informal Basque      (column: Ref-ISMD-Basque(ORIGINAL))
"""


def build_index(examples: list[dict], encoder, src_key: str = "input"):
    """
    Encodes examples and builds a FAISS index for nearest-neighbour search.

    Suggested approach (by Claude):

        import faiss
        import numpy as np

        texts = [ex[src_key] for ex in examples]       # extract source texts
        vectors = encoder.encode(texts)                 # encode them — shape: (n, embedding_dim)
        vectors = np.array(vectors).astype("float32")   # FAISS needs float32

        index = faiss.IndexFlatL2(vectors.shape[1])     # flat L2 index (simple and reliable)
        index.add(vectors)                              # add vectors to the index
        return index

    For the encoder, something like this should work well for Spanish and Basque:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    This function is then used 4 times in scripts/build_indexes.py to build the four indexes we need.
    """
    raise NotImplementedError


def retrieve(query: str, k: int, index, examples: list[dict], encoder) -> list[dict]:
    """
    Finds the k training examples most similar to the query in an index.

    Suggested approach:

        import numpy as np

        vec = encoder.encode([query])                   # encode the query — shape: (1, embedding_dim)
        vec = np.array(vec).astype("float32")           # FAISS needs float32
        _, idxs = index.search(vec, k)                  # search — idxs shape: (1, k)
        return [examples[i] for i in idxs[0]]           # return the matching examples
    """
    raise NotImplementedError
