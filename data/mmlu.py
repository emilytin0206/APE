# data/mmlu.py
from datasets import load_dataset

def load_mmlu_data(subset: str, split: str = 'test', limit: int = None):
    """
    Loads MMLU data and returns it in APE format: (inputs, outputs)
    inputs: List of question strings (formatted with options).
    outputs: List of LISTS of valid answers (e.g., [['A'], ['B']]).
    """
    print(f"Loading MMLU: {subset} ({split})...")
    dataset = load_dataset("cais/mmlu", subset, split=split)
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        
    inputs = []
    outputs = []
    
    answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    for ex in dataset:
        # Format: Question + Options
        q = f"Question: {ex['question']}\nOptions:\n"
        for i, opt in enumerate(ex['choices']):
            q += f"{answer_map[i]}. {opt}\n"
        q += "Answer:" # Prompt ending
        
        # Format: Answer (Must be list of strings for APE utility)
        ans = answer_map[ex['answer']]
        
        inputs.append(q.strip())
        outputs.append([ans]) # Wrap in list!
        
    return inputs, outputs