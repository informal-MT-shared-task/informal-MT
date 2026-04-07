"""
Evaluation utilities for chrF++.
"""

from sacrebleu.metrics import CHRF


def compute_chrf_pp(hypotheses: list, references: list) -> float:
    """
    Compute the corpus-level chrF++ score.

    Both inputs should contain one sentence per element.

    Returns:
        chrF++ score as a float (0–100).
    """
    assert len(hypotheses) == len(references), (
        f"Length mismatch: {len(hypotheses)} hypotheses vs {len(references)} references"
    )
    metric = CHRF(word_order=2)   
    result = metric.corpus_score(hypotheses, [references])
    
    return result.score


def evaluate_file(hyp_path: str, ref_path: str) -> float:
    """
    Read hypotheses and references from text files
    (one sentence per line) and compute chrF++.
    """
    
    with open(hyp_path, encoding="utf-8") as fh:
        hypotheses = [line.strip() for line in fh if line.strip()]
    with open(ref_path, encoding="utf-8") as fh:
        references = [line.strip() for line in fh if line.strip()]
    score = compute_chrf_pp(hypotheses, references)
    print(f"chrF++ score: {score:.2f}")
    
    return score
