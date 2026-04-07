def format_few_shot_examples(examples: list) -> str:
    """
    Formats few-shot pairs.
    """
    if not examples:
        return ""
    
    blocks = []
    for ex in examples:
        source = ex.get('input', '')
        target = ex.get('output', '')
        
        blocks.append(f"Input: {source}\nOutput: {target}")
    
    delimiter = "\n\n###\n\n"
    return delimiter.join(blocks)

def prompt_builder(prompt, few_shot_examples):
    final_prompt = prompt + format_few_shot_examples(few_shot_examples)
    return final_prompt