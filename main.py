"""
Entry point.

Usage examples:
    python main.py --approach one_step --k 5
    python main.py --approach multi_step --k 3
    python main.py --approach one_step --eval-only --run-name my_run
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
from src.rag.retriever import build_index, retrieve

OUTPUTS_DIR = Path("outputs")
PROMPTS_YAML = Path("config/prompts.yaml")
EXPERIMENT_CONFIG_YAML = Path("config/experiment_config.yaml")


def load_configs(config_path: str):
    with open(config_path) as f:
        exp_cfg = yaml.safe_load(f)
    with open(PROMPTS_YAML) as f:
        prompts_cfg = yaml.safe_load(f)
    return exp_cfg, prompts_cfg


def save_outputs(hypotheses: list[str], references: list[str], run_name: str):
    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hypotheses.txt").write_text("\n".join(hypotheses), encoding="utf-8")
    (out_dir / "references.txt").write_text("\n".join(references), encoding="utf-8")


def run_eval_only(run_name: str):
    out_dir = OUTPUTS_DIR / run_name
    hyp_file = out_dir / "hypotheses.txt"
    ref_file = out_dir / "references.txt"
    if not hyp_file.exists() or not ref_file.exists():
        raise FileNotFoundError(
            f"No hypotheses/references found for run '{run_name}'. "
            "Run translation first (without --eval-only)."
        )
    score = evaluate_file(str(hyp_file), str(ref_file))
    (out_dir / "scores.json").write_text(
        json.dumps({"chrf++": score, "run": run_name}, indent=2)
    )


def run_one_step(exp_cfg: dict, prompts_cfg: dict, run_name: str, k: int):
    cfg = exp_cfg["one_step"]
    test = load_tsv(exp_cfg["data"]["tsv_path"])

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

    save_outputs(hypotheses, references, run_name)
    out_dir = OUTPUTS_DIR / run_name
    score = evaluate_file(str(out_dir / "hypotheses.txt"), str(out_dir / "references.txt"))
    (out_dir / "scores.json").write_text(
        json.dumps({"chrf++": score, "run": run_name, "approach": "one_step"}, indent=2)
    )


def run_multi_step(exp_cfg: dict, prompts_cfg: dict, run_name: str, k: int):
    cfg = exp_cfg["multi_step"]
    test = load_tsv(exp_cfg["data"]["tsv_path"])

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

    save_outputs(hypotheses, references, run_name)
    out_dir = OUTPUTS_DIR / run_name
    score = evaluate_file(str(out_dir / "hypotheses.txt"), str(out_dir / "references.txt"))
    (out_dir / "scores.json").write_text(
        json.dumps({"chrf++": score, "run": run_name, "approach": "multi_step"}, indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description="Informal Spanish→Basque MT pipeline")
    parser.add_argument("--approach", choices=["one_step", "multi_step"], default="one_step",
                        help="Which pipeline to run (default: one_step)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of few-shot examples to retrieve (default: 5)")
    parser.add_argument("--config", default=str(EXPERIMENT_CONFIG_YAML),
                        help="Path to experiment_config.yaml")
    parser.add_argument("--run-name", default=None,
                        help="Name for the output directory (default: one_step/multi_step)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip translation; just (re)score saved hypotheses")
    args = parser.parse_args()

    run_name = args.run_name or args.approach

    if args.eval_only:
        run_eval_only(run_name)
        return

    exp_cfg, prompts_cfg = load_configs(args.config)

    if args.approach == "one_step":
        run_one_step(exp_cfg, prompts_cfg, run_name, args.k)
    else:
        run_multi_step(exp_cfg, prompts_cfg, run_name, args.k)


if __name__ == "__main__":
    main()
