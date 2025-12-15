# ape/generate.py
import random
from typing import List, Tuple
from .llm import LLM

def generate_prompts(
    model: LLM,
    data: Tuple[List[str], List[List[str]]],
    config: dict
) -> List[str]:
    """
    Generates candidate prompts using the APE template strategy.
    """
    prompt_gen_template = config.get('prompt_gen_template')
    demos_template = config.get('demos_template')
    num_demos = config.get('num_demos', 3)
    num_subsamples = config.get('num_candidates', 5) # Number of queries to send
    
    inputs, outputs = data
    queries = []

    for _ in range(num_subsamples):
        # Randomly sample demonstrations
        indices = random.sample(range(len(inputs)), min(num_demos, len(inputs)))
        
        demo_strs = []
        for i in indices:
            d = demos_template.replace('[INPUT]', inputs[i])\
                              .replace('[OUTPUT]', outputs[i][0])
            demo_strs.append(d)
        
        full_demo = "\n\n".join(demo_strs)
        
        # Fill generation template
        # Note: APE generation template usually puts the target input/output at the end too
        # But simplistic implementation often just asks LLM to deduce instruction from demos.
        query = prompt_gen_template.replace('[full_DEMO]', full_demo)
        queries.append(query)

    print(f"Generating candidates using {model.model_name}...")
    candidates = model.generate(queries)
    
    # Deduplicate and clean
    unique_candidates = list(set([c.strip() for c in candidates if c.strip()]))
    return unique_candidates