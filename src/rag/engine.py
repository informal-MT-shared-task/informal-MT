"""
Retrieval strategies for few-shot example selection.

PhenomenaRetriever — selects examples based on informality phenomena detected
in the query, using ground-truth annotation columns from training examples.

Usage
-----
from src.rag.retriever import load_encoder, load_retriever_fn
from src.rag.engine import PhenomenaRetriever
from src.translation.utils import load_tsv

samples = load_tsv("data/train.tsv")
cfg = experiment_config["rag"]

retriever = PhenomenaRetriever(samples, cfg)
retriever_fn = retriever.as_retriever_fn()   # -> Callable[[str, int], list[dict]]
"""

from __future__ import annotations

import random
import warnings
from typing import Callable

from src.rag.phenomena import detect_phenomena, lexical_overlap


class PhenomenaRetriever:
    """
    Retrieves k few-shot examples using a two-stage strategy:

    Slots 1-3 (fixed): always the same hardcoded examples — cover all core phenomena.
    Slots 4-k (dynamic): filled based on phenomena detected in the query.

    Parameters
    ----------
    samples:
        Full list of Sample objects from the training pool.
    config:
        The ``rag`` block from experiment_config.yaml. Must contain:
            hardcoded_example_indices: list[int]  (at least 12 indices)
            n_examples: int
    """

    def __init__(self, samples: list, config: dict):
        self.samples = samples
        self.n_examples = config.get("n_examples", 12)

        indices = config.get("hardcoded_example_indices", [])
        if not indices:
            warnings.warn(
                "PhenomenaRetriever: hardcoded_example_indices is empty in experiment_config.yaml. "
                "Please fill it with training example indices before running. "
                "Falling back to first n_examples samples.",
                UserWarning,
            )
            indices = list(range(min(self.n_examples, len(samples))))

        if len(indices) < 3:
            raise ValueError(
                f"PhenomenaRetriever needs at least 3 hardcoded_example_indices (fixed anchor slots), "
                f"got {len(indices)}."
            )

        # Fixed slots scale proportionally with n_examples (same ratio as baseline 3/12 = 1/4)
        n_fixed = max(1, self.n_examples // 4)
        self.fixed_indices = indices[:n_fixed]
        self.fallback_indices = indices[n_fixed:]  # used as fallback when no phenomena detected

    def _sample_to_dict(self, sample) -> dict:
        return {"input": sample.source_es, "output": sample.ref_informal_eu}

    def retrieve(self, query: str, k: int) -> list[dict]:
        """
        Return k examples for query.

        Always starts with the 3 fixed examples, then fills remaining slots
        dynamically based on detected phenomena.
        """
        k = min(k, self.n_examples)
        fixed = [self.samples[i] for i in self.fixed_indices]
        fixed_set = set(self.fixed_indices)

        n_dynamic = k - len(self.fixed_indices)
        if n_dynamic <= 0:
            return [self._sample_to_dict(s) for s in fixed[:k]]

        phenomena = detect_phenomena(query)
        detected = [p for p, v in phenomena.items() if v == 1]

        if len(detected) == 0:
            # No phenomena: use fallback hardcoded examples
            dynamic_samples = [
                self.samples[i] for i in self.fallback_indices
                if i not in fixed_set
            ][:n_dynamic]

        elif len(detected) == 1:
            # One phenomenon: top n_dynamic by lexical overlap from matching pool
            phenomenon = detected[0]
            pool = self._get_phenomenon_pool(phenomenon, exclude=fixed_set)
            pool_sorted = sorted(pool, key=lambda s: lexical_overlap(query, s.source_es), reverse=True)
            dynamic_samples = pool_sorted[:n_dynamic]

            # Fill remaining slots randomly if pool too small
            if len(dynamic_samples) < n_dynamic:
                used = fixed_set | {self.samples.index(s) for s in dynamic_samples}
                dynamic_samples += self._random_fill(n_dynamic - len(dynamic_samples), exclude=used)

        else:
            # Two phenomena: split slots between them, larger pool gets the extra slot
            p_a, p_b = detected[0], detected[1]
            pool_a = self._get_phenomenon_pool(p_a, exclude=fixed_set)
            pool_b = self._get_phenomenon_pool(p_b, exclude=fixed_set)

            n_a, n_b = (n_dynamic // 2, n_dynamic - n_dynamic // 2)
            if len(pool_a) < len(pool_b):
                n_a, n_b = n_b, n_a  # larger pool gets the bigger share

            pool_a_sorted = sorted(pool_a, key=lambda s: lexical_overlap(query, s.source_es), reverse=True)
            pool_b_sorted = sorted(pool_b, key=lambda s: lexical_overlap(query, s.source_es), reverse=True)

            picked_a = pool_a_sorted[:n_a]
            picked_a_set = fixed_set | {self._index_of(s) for s in picked_a}
            picked_b = [s for s in pool_b_sorted if self._index_of(s) not in picked_a_set][:n_b]

            dynamic_samples = picked_a + picked_b

            # Fill remaining slots randomly if pools too small
            if len(dynamic_samples) < n_dynamic:
                used = fixed_set | {self._index_of(s) for s in dynamic_samples}
                dynamic_samples += self._random_fill(n_dynamic - len(dynamic_samples), exclude=used)

        result = fixed + dynamic_samples
        return [self._sample_to_dict(s) for s in result[:k]]

    def _get_phenomenon_pool(self, phenomenon: str, exclude: set) -> list:
        """Return training samples that have the given phenomenon (using annotation columns)."""
        def has_phenomenon(sample) -> bool:
            if phenomenon == "elongation":
                try:
                    return float(sample.phonetic_stylization) > 0
                except (ValueError, TypeError):
                    return False
            if phenomenon == "informal_lexical":
                try:
                    return float(sample.informal_lexical_item) > 0
                except (ValueError, TypeError):
                    return False
            return False

        return [
            s for i, s in enumerate(self.samples)
            if i not in exclude and has_phenomenon(s)
        ]

    def _index_of(self, sample) -> int:
        return self.samples.index(sample)

    def _random_fill(self, n: int, exclude: set) -> list:
        pool = [s for i, s in enumerate(self.samples) if i not in exclude]
        return random.sample(pool, min(n, len(pool)))

    def as_retriever_fn(self) -> Callable[[str, int], list[dict]]:
        """Return a retriever_fn compatible with the pipeline: (query, k) -> list[dict]."""
        return lambda query, k: self.retrieve(query, k)
