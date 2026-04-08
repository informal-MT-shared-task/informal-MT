"""
Entry point.

Usage examples:
    python main.py --approach one_step --k 5
    python main.py --approach multi_step --k 3
    python main.py --approach one_step --eval-only --k 5
    python main.py --approach one_step --config config/experiment_config.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

from src.translation.utils import load_tsv, translate_batch
from src.translation.pipeline import InformalSpanishToInformalBasque, MultiStepApproach
from src.translation.llm import load_latxa, load_llama3
from src.evaluation.metrics import evaluate_file
from utils import load_configs, save_outputs

OUTPUTS_DIR = Path("outputs")
PROMPTS_YAML = Path("config/prompts.yaml")
EXPERIMENT_CONFIG_YAML = Path("config/experiment_config.yaml")


def run_eval_only(k: int):
    matches = list(OUTPUTS_DIR.glob(f"*_{k}-shot_hypotheses.txt"))
    if not matches:
        raise FileNotFoundError(
            f"No hypothesis file found for k={k}. Run translation first (without --eval-only)."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple hypothesis files found for k={k}: {[f.name for f in matches]}. "
            "Delete the one you don't want to score."
        )
    hyp_file = matches[0]
    approach = hyp_file.stem.replace(f"_{k}-shot_hypotheses", "")
    ref_file = OUTPUTS_DIR / "references.txt"
    if not ref_file.exists():
        raise FileNotFoundError("references.txt not found. Run translation first (without --eval-only).")
    score = evaluate_file(str(hyp_file), str(ref_file))
    (OUTPUTS_DIR / f"{approach}_{k}-shot_scores.json").write_text(
        json.dumps({"chrf++": score, "approach": approach, "k": k}, indent=2)
    )


def run_one_step(exp_cfg: dict, prompts_cfg: dict, k: int):
    cfg = exp_cfg["one_step"]
    test = load_tsv(exp_cfg["data"]["test_tsv_path"])

    print(f"Test set size: {len(test)}")
    print(f"Loading model: {cfg['model']} ...")

    if cfg["model"] == "latxa":
        tokenizer, model = load_latxa()
    else:
        tokenizer, model = load_llama3()

    # TODO: when retriever is ready, replace these with real examples and a real encoder
    # examples_step0 = [{"input": ..., "output": ...}, ...]
    # encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    # index_step0 = build_index(examples_step0, encoder)
    # retriever_fn_step0 = lambda query, k: retrieve(query, k, index_step0, examples_step0, encoder)
    retriever_fn_step0 = lambda query, k: []  # placeholder — returns no examples until RAG is ready

    pipeline = InformalSpanishToInformalBasque(tokenizer, model, retriever_fn_step0, cfg, prompts_cfg)

    source_texts = [s.source_es for s in test]
    references = [getattr(s, cfg["ref_column"]) for s in test]

    hypotheses = translate_batch(source_texts, pipeline.translate_informal_spanish_to_informal_basque, k)

    save_outputs(hypotheses, references, "one_step", k, OUTPUTS_DIR)
    score = evaluate_file(str(OUTPUTS_DIR / f"one_step_{k}-shot_hypotheses.txt"), str(OUTPUTS_DIR / "references.txt"))
    (OUTPUTS_DIR / f"one_step_{k}-shot_scores.json").write_text(
        json.dumps({"chrf++": score, "approach": "one_step", "k": k}, indent=2)
    )


def run_multi_step(exp_cfg: dict, prompts_cfg: dict, k: int):
    cfg = exp_cfg["multi_step"]
    test = load_tsv(exp_cfg["data"]["test_tsv_path"])

    print(f"Test set size: {len(test)}")
    print("Loading models ...")

    norm_tokenizer, norm_model = load_llama3()   # Step 1
    mt_tokenizer, mt_model = load_latxa()        # Steps 2 & 3

    # TODO: replace these with real examples and a real encoder, one pool per step
    # encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    # index_step1 = build_index(examples_step1, encoder)
    # retriever_fn_step1 = lambda query, k: retrieve(query, k, index_step1, examples_step1, encoder)
    # (same for step2 and step3)
    retriever_fn_step1 = lambda query, k: []  # placeholder
    retriever_fn_step2 = lambda query, k: []  # placeholder
    retriever_fn_step3 = lambda query, k: []  # placeholder

    pipeline = MultiStepApproach(
        norm_tokenizer, norm_model,
        mt_tokenizer, mt_model,
        retriever_fn_step1, retriever_fn_step2, retriever_fn_step3,
        cfg, prompts_cfg,
    )

    source_texts = [s.source_es for s in test]
    references = [getattr(s, cfg["ref_column"]) for s in test]

    hypotheses = translate_batch(source_texts, pipeline.translate_multi_step, k)

    save_outputs(hypotheses, references, "multi_step", k, OUTPUTS_DIR)
    score = evaluate_file(str(OUTPUTS_DIR / f"multi_step_{k}-shot_hypotheses.txt"), str(OUTPUTS_DIR / "references.txt"))
    (OUTPUTS_DIR / f"multi_step_{k}-shot_scores.json").write_text(
        json.dumps({"chrf++": score, "approach": "multi_step", "k": k}, indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description="Informal Spanish→Basque MT pipeline")
    parser.add_argument("--approach", choices=["one_step", "multi_step"], default="one_step",
                        help="Which pipeline to run (default: one_step)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of few-shot examples to retrieve (default: 5)")
    parser.add_argument("--config", default=str(EXPERIMENT_CONFIG_YAML),
                        help="Path to experiment_config.yaml")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip translation; just (re)score saved hypotheses")
    args = parser.parse_args()

    if args.eval_only:
        run_eval_only(args.k)
        return

    exp_cfg, prompts_cfg = load_configs(args.config, PROMPTS_YAML)

    if args.approach == "one_step":
        run_one_step(exp_cfg, prompts_cfg, args.k)
    else:
        run_multi_step(exp_cfg, prompts_cfg, args.k)


if __name__ == "__main__":
    main()
