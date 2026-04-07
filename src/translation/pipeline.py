"""
Two translation pipelines:

InformalSpanishToInformalBasque: direct few-shot:
Informal Spanish → Informal Basque

MultiStepApproach: 3-steps:
    Step 1: Informal Spanish  → Standard Spanish   (Llama 3, normalization)
    Step 2: Standard Spanish  → Standard Basque    (Latxa, standard MT)
    Step 3: Standard Basque   → Informal Basque    (Latxa, style injection)

All steps use few-shot retrieval. Each retriever is a plain function: retriever_fn(query, k) -> list[dict].
See main.py for how to build and pass these functions.
"""

from src.translation.utils import translate, translate_batch


class InformalSpanishToInformalBasque:
    def __init__(self, tokenizer, model, retriever_fn_step0, config: dict, prompts_config: dict):
        """
        tokenizer, model   : loaded model instance
        retriever_fn_step0 : function(query, k) -> list[dict], retrieves informal ES → informal EU examples
        config             : experiment_config["one_step"]
        prompts_config     : full prompts.yaml dict
        """
        self.tokenizer = tokenizer
        self.model = model
        self.retriever_fn_step0 = retriever_fn_step0
        self.config = config
        self.prompts_config = prompts_config

    def translate_informal_spanish_to_informal_basque(self, source_text: str, k: int) -> str:
        return translate(
            source_text=source_text,
            k=k,
            retriever_fn=self.retriever_fn_step0,
            tokenizer=self.tokenizer,
            model=self.model,
            prompt_config=self.prompts_config["one_step"],
            gen_config=self.config["generation"],
        )


class MultiStepApproach:
    def __init__(
        self,
        norm_tokenizer, norm_model,     # Step 1: Llama 3
        mt_tokenizer,   mt_model,       # Steps 2 and 3: Latxa
        retriever_fn_step1,             # function(query, k) — informal ES → standard ES examples
        retriever_fn_step2,             # function(query, k) — standard ES → standard EU examples
        retriever_fn_step3,             # function(query, k) — standard EU → informal EU examples
        config: dict,
        prompts_config: dict,
    ):
        """
        norm_tokenizer/norm_model   : model used for Step 1
        mt_tokenizer/mt_model       : model used for Steps 2 and 3
        retriever_fn_step1/2/3      : one retriever function per step, each with its own example pool
        config                      : experiment_config["multi_step"]
        prompts_config              : full prompts.yaml dict
        """
        self.norm_tok = norm_tokenizer
        self.norm_model = norm_model
        self.mt_tok = mt_tokenizer
        self.mt_model = mt_model
        self.retriever_fn_step1 = retriever_fn_step1
        self.retriever_fn_step2 = retriever_fn_step2
        self.retriever_fn_step3 = retriever_fn_step3
        self.config = config
        self.prompts_config = prompts_config

    def translate_multi_step(self, source_text: str, k: int) -> str:
        normalized_es = translate(
            source_text=source_text,
            k=k,
            retriever_fn=self.retriever_fn_step1,
            tokenizer=self.norm_tok,
            model=self.norm_model,
            prompt_config=self.prompts_config["multi_step"]["step1"],
            gen_config=self.config["step1"]["generation"],
        )
        standard_eu = translate(
            source_text=normalized_es,
            k=k,
            retriever_fn=self.retriever_fn_step2,
            tokenizer=self.mt_tok,
            model=self.mt_model,
            prompt_config=self.prompts_config["multi_step"]["step2"],
            gen_config=self.config["step2"]["generation"],
        )
        informal_eu = translate(
            source_text=standard_eu,
            k=k,
            retriever_fn=self.retriever_fn_step3,
            tokenizer=self.mt_tok,
            model=self.mt_model,
            prompt_config=self.prompts_config["multi_step"]["step3"],
            gen_config=self.config["step3"]["generation"],
        )
        return informal_eu
