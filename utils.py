import yaml


def load_configs(config_path: str, prompts_path):
    with open(config_path) as f:
        exp_cfg = yaml.safe_load(f)
    with open(prompts_path) as f:
        prompts_cfg = yaml.safe_load(f)
    return exp_cfg, prompts_cfg 


def save_outputs(hypotheses: list[str], references: list[str], approach: str, k: int, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned = [h.replace("\n", " ").strip() for h in hypotheses]
    (out_dir / f"{approach}_{k}-shot_hypotheses.txt").write_text("\n".join(cleaned), encoding="utf-8")
    (out_dir / "references.txt").write_text("\n".join(references), encoding="utf-8")