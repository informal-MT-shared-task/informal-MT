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
from src.rag.retriever import load_encoder, load_retriever_fn
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


def run_one_step(exp_cfg: dict, prompts_cfg: dict, k: int, use_rag: bool = False):
    cfg = exp_cfg["one_step"]
    test = load_tsv(exp_cfg["data"]["test_tsv_path"])

    print(f"Test set size: {len(test)}")
    print(f"Loading model: {cfg['model']} ...")

    if cfg["model"] == "latxa":
        tokenizer, model = load_latxa()
    else:
        tokenizer, model = load_llama3()

    if use_rag:
        print("Loading encoder and retriever ...")
        encoder = load_encoder()
        samples = load_tsv(exp_cfg["data"]["tsv_path"])
        examples_step0 = [{"input": s.source_es, "output": s.ref_informal_eu} for s in samples]
        retriever_fn_step0 = load_retriever_fn("data/index_step0.faiss", examples_step0, encoder)
    else:
        # Hardcoded examples from train.tsv — no overlap with test
        # Selected for diversity: question preservation, no hallucination, phonetic elongation,
        # code-switching, dialect features, informal lexic
        hardcoded_examples = [
            # --- targeted: question preservation ---
            {"input": "tu pq ??",                                                                 "output": "Zuk zeba ??"},
            {"input": "como lo sabeees?? yo tengo turno  luego a las 9 ehh",                     "output": "Zelan dakizu!!? nik gero 9tan dakot turnue"},
            {"input": "madreeeee q miedo... q estais haciendo???",                               "output": "Amaaaa ze bildurre...zertan zaizte???"},
            # --- targeted: no hallucination ---
            {"input": "pero estoy con alain y en nada se va",                                    "output": "Baina alainekin nago ta laister jungo da"},
            {"input": "no puedo escuchar estoy en la biblioteca",                                "output": "Ezin dut entzun liburutegian naiiiz"},
            # --- targeted: simple informal correctness ---
            {"input": "siiiii es buena ideaaa",                                                  "output": "Baiiii ideia ona daaaa"},
            {"input": "conseguidooo",                                                            "output": "Lortu duuut"},
            # --- phonetic elongation ---
            {"input": "yaaa jajajaj ya te dije",                                                 "output": "Yaaa jajajaj esantzutenn"},
            {"input": "A ver que pasaaaaaaa",                                                    "output": "Ea zer gertatzen deeeeen"},
            # --- code-switching preserved ---
            {"input": "no te pega llorar",                                                       "output": "No te pega negar iteaa"},
            {"input": "eii menos con ane",                                                       "output": "Eii menos anetxokin"},
            {"input": "buenooo... tus gustos...",                                                "output": "Buenooo.... zure gustok..."},
            {"input": "El lunes tenemos q hacer terapia q estoy fatall",                         "output": "Astelehenenn terapia in behar deuu que estoy fatall"},
            # --- dialect + phonetic ---
            {"input": "ella tampoco las tiene rectas eh",                                        "output": "Berak be eztakoz rektoo"},
            {"input": "a ver si le conocemos en alguna juerga",                                  "output": "Abeer juerganbaten ezautzen deun"},
            # --- remaining (used only with k > 15) ---
            {"input": "es q es un poco tontillo no se enterara!",                               "output": "Esk tonto xamarra da eztaa kontuauko!"},
            {"input": "ahora voy a comer y luego siestita y libre",                              "output": "Nik orain bazkalduko dut ta gero siesta pixkat ta libreee"},
            {"input": "tio araan entre esto y los bertsos de tu abu me emociono mogollon!!",     "output": "Tioo araan hau ta zure aitonan bertsokin mordoa emozionatzen naiz!??"},
            {"input": "ajjajaja ya ya tampoco tengo tan mal gusto jajajaj",                      "output": "Ajjajaja ya ya....  Eztakot hain gusto txarra jajajaj"},
            {"input": "waaaa es en tu casa??",                                                   "output": "Waaaa zure etxean da??"},
        ]
        retriever_fn_step0 = lambda query, k: hardcoded_examples[:k]

    pipeline = InformalSpanishToInformalBasque(tokenizer, model, retriever_fn_step0, cfg, prompts_cfg)

    source_texts = [s.source_es for s in test]
    references = [getattr(s, cfg["ref_column"]) for s in test]

    hypotheses = translate_batch(source_texts, pipeline.translate_informal_spanish_to_informal_basque, k)

    save_outputs(hypotheses, references, "one_step", k, OUTPUTS_DIR)
    score = evaluate_file(str(OUTPUTS_DIR / f"one_step_{k}-shot_hypotheses.txt"), str(OUTPUTS_DIR / "references.txt"))
    (OUTPUTS_DIR / f"one_step_{k}-shot_scores.json").write_text(
        json.dumps({"chrf++": score, "approach": "one_step", "k": k}, indent=2)
    )


def run_multi_step(exp_cfg: dict, prompts_cfg: dict, k: int, use_rag: bool = False):
    cfg = exp_cfg["multi_step"]
    test = load_tsv(exp_cfg["data"]["test_tsv_path"])
 
    print(f"Test set size: {len(test)}")
    print("Loading models ...")
 
    norm_tokenizer, norm_model = load_llama3()   # Step 1
    mt_tokenizer, mt_model = load_latxa()        # Steps 2 & 3
 
    print("Loading encoder and retrievers ...")
    encoder = load_encoder()
 
    # Step 1 — normalization: informal ES → standard ES  (needs dataset_augmented.tsv)
    samples_aug = load_tsv(exp_cfg["data"]["augmented_tsv_path"])
    examples_step1 = [{"input": s.source_es,    "output": s.normalized_es}   for s in samples_aug]
    examples_step2 = [{"input": s.normalized_es, "output": s.ref_formal_eu}  for s in samples_aug]
    retriever_fn_step1 = load_retriever_fn("data/index_step1.faiss", examples_step1, encoder)
    retriever_fn_step2 = load_retriever_fn("data/index_step2.faiss", examples_step2, encoder)
 
    # Step 3 — style injection: standard EU → informal EU  (main dataset)
    samples = load_tsv(exp_cfg["data"]["tsv_path"])
    examples_step3 = [{"input": s.ref_formal_eu, "output": s.ref_informal_eu} for s in samples]
    retriever_fn_step3 = load_retriever_fn("data/index_step3.faiss", examples_step3, encoder)


    # # TODO: replace these with real examples and a real encoder, one pool per step
    # # encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    # # index_step1 = build_index(examples_step1, encoder)
    # # retriever_fn_step1 = lambda query, k: retrieve(query, k, index_step1, examples_step1, encoder)
    # # (same for step2 and step3)
    # retriever_fn_step1 = lambda query, k: []  # placeholder
    # retriever_fn_step2 = lambda query, k: []  # placeholder
    # retriever_fn_step3 = lambda query, k: []  # placeholder

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
    parser.add_argument("--rag", action="store_true",
                        help="Use FAISS retriever instead of hardcoded examples")
    args = parser.parse_args()

    if args.eval_only:
        run_eval_only(args.k)
        return

    exp_cfg, prompts_cfg = load_configs(args.config, PROMPTS_YAML)

    if args.approach == "one_step":
        run_one_step(exp_cfg, prompts_cfg, args.k, use_rag=args.rag)
    else:
        run_multi_step(exp_cfg, prompts_cfg, args.k, use_rag=args.rag)


if __name__ == "__main__":
    main()
