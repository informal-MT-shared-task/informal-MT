"""
src/rag/retriever.py

FAISS-based retriever for few-shot example lookup.


load_encoder(model_name)                        -> SentenceTransformer
build_index(examples, encoder, src_key)         -> faiss.IndexFlatL2
retrieve(query, k, index, examples, encoder)    -> list[dict]
make_retriever_fn(index, examples, encoder)     -> Callable[[str, int], list[dict]]
load_retriever_fn(index_path, examples, encoder)-> Callable[[str, int], list[dict]]
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_ENCODER_MODEL = "paraphrase-multilingual-mpnet-base-v2"


def load_encoder(model_name: str = DEFAULT_ENCODER_MODEL) -> SentenceTransformer:
    """Load a SentenceTransformer encoder by name."""
    return SentenceTransformer(model_name)


def build_index(
    examples: list[dict],
    encoder: SentenceTransformer,
    src_key: str = "input",
) -> faiss.IndexFlatL2:
    """
    Encode the source texts of ``examples`` and build a FAISS flat-L2 index.

    Parameters
    ----------
    examples:
        List of dicts with at least ``src_key`` and ``"output"`` fields.
    encoder:
        SentenceTransformer instance used to encode the texts.
    src_key:
        Dict key whose value is encoded (default: ``"input"``).

    Returns
    -------
    faiss.IndexFlatL2
        Populated index. Keep ``examples`` alongside it to map hits back to dicts.
    """
    texts = [ex[src_key] for ex in examples]
    vectors = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    vectors = np.array(vectors).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(
    query: str,
    k: int,
    index: faiss.IndexFlatL2,
    examples: list[dict],
    encoder: SentenceTransformer,
) -> list[dict]:
    """
    Return the ``k`` most similar examples to ``query``.

    Parameters
    ----------
    query:
        Source text to look up.
    k:
        Number of nearest neighbours to return.
    index:
        FAISS index built with ``build_index()``.
    examples:
        Same list of dicts passed to ``build_index()``.
    encoder:
        Same encoder used when building the index.

    Returns
    -------
    list[dict]
        Up to ``k`` dicts with ``"input"`` / ``"output"`` keys, ordered by similarity.
    """
    vec = encoder.encode([query], convert_to_numpy=True)
    vec = np.array(vec).astype("float32")
    k = min(k, index.ntotal)
    _, idxs = index.search(vec, k)
    return [examples[i] for i in idxs[0]]


def make_retriever_fn(
    index: faiss.IndexFlatL2,
    examples: list[dict],
    encoder: SentenceTransformer,
):
    """
    Wrap ``(index, examples, encoder)`` into the callable expected by the pipelines:

        retriever_fn(query: str, k: int) -> list[dict]
    """
    def retriever_fn(query: str, k: int) -> list[dict]:
        return retrieve(query, k, index, examples, encoder)
    return retriever_fn


def load_retriever_fn(
    index_path: str | Path,
    examples: list[dict],
    encoder: SentenceTransformer,
):
    """
    Load a saved FAISS index from disk and return a ready-to-use retriever_fn.

    Parameters
    ----------
    index_path:
        Path to a ``.faiss`` file saved with ``faiss.write_index()``.
    examples:
        The same list of dicts used when the index was built.
    encoder:
        The same encoder used when the index was built.

    Usage (in main.py)
    ------------------
    from src.rag.retriever import load_encoder, load_retriever_fn
    from src.translation.utils import load_tsv

    encoder = load_encoder()
    samples = load_tsv("data/dataset.tsv")

    examples_step0 = [{"input": s.source_es, "output": s.ref_informal_eu} for s in samples]
    retriever_fn_step0 = load_retriever_fn("data/index_step0.faiss", examples_step0, encoder)

    examples_step3 = [{"input": s.ref_formal_eu, "output": s.ref_informal_eu} for s in samples]
    retriever_fn_step3 = load_retriever_fn("data/index_step3.faiss", examples_step3, encoder)
    """
    index = faiss.read_index(str(index_path))
    return make_retriever_fn(index, examples, encoder)