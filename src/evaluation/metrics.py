"""
Evaluation utilities for chrF++ and BLEU.
"""
from sacrebleu.metrics import CHRF, BLEU

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


def compute_bleu(hypotheses: list, references: list) -> float:
    """
    Compute the corpus-level BLEU score.
    Both inputs should contain one sentence per element.
    Returns:
        BLEU score as a float (0–100).
    """
    assert len(hypotheses) == len(references), (
        f"Length mismatch: {len(hypotheses)} hypotheses vs {len(references)} references"
    )
    metric = BLEU(effective_order=True)
    result = metric.corpus_score(hypotheses, [references])
    return result.score


def evaluate_file(hyp_path: str, ref_path: str) -> dict:
    """
    Read hypotheses and references from text files
    (one sentence per line) and compute chrF++ and BLEU.
    Returns:
        dict with keys 'chrf++' and 'bleu'.
    """
    with open(hyp_path, encoding="utf-8") as fh:
        hypotheses = [line.strip() for line in fh if line.strip()]
    with open(ref_path, encoding="utf-8") as fh:
        references = [line.strip() for line in fh if line.strip()]

    chrf_score = compute_chrf_pp(hypotheses, references)
    bleu_score = compute_bleu(hypotheses, references)

    print(f"chrF++ score: {chrf_score:.2f}")
    print(f"BLEU   score: {bleu_score:.2f}")

    return {"chrf++": chrf_score, "bleu": bleu_score}