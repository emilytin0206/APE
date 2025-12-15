# ape/evaluator.py
import numpy as np
from typing import List, Tuple, Dict, Any
from .llm import LLM
from . import utility

def get_query(prompt, eval_template, input_, demos_template, demo_data):
    """Constructs the final prompt string including few-shot demos."""
    # 建立 Few-shot 範例字串
    demo_strs = []
    if demo_data:
        inputs, outputs = demo_data
        for inp, out in zip(inputs, outputs):
            # output 是一個 list，取第一個作為示範
            d = demos_template.replace('[INPUT]', inp).replace('[OUTPUT]', out[0])
            demo_strs.append(d)
    full_demo = "\n\n".join(demo_strs)

    # 填入主模板
    query = eval_template.replace('[PROMPT]', prompt)\
                         .replace('[INPUT]', input_)\
                         .replace('[full_DEMO]', full_demo)\
                         .replace('[OUTPUT]', '') # 留空給模型回答
    return query

def exec_accuracy_evaluator(
    model: LLM,
    prompts: List[str],
    eval_data: Tuple[List[str], List[List[str]]],
    few_shot_data: Tuple[List[str], List[List[str]]],
    config: Dict[str, Any]
) -> List[Tuple[str, float]]:
    
    # Config parameters
    num_samples = config.get('num_samples', 20)
    num_few_shot = config.get('num_few_shot', 0)
    eval_template = config.get('eval_template', "Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]")
    demos_template = config.get('demos_template', "Input: [INPUT]\nOutput: [OUTPUT]")

    # Subsample evaluation data (Logic identical to APE)
    indices = list(range(len(eval_data[0])))
    # Fixed subsample for fair comparison across prompts
    sample_indices = indices[:min(num_samples, len(indices))]
    
    queries = []
    ground_truths = []
    
    # 為了保持官方邏輯，這裡展開 Loop
    # 官方代碼是對每個 prompt 重新採樣 few-shot data
    import random
    
    for prompt in prompts:
        for idx in sample_indices:
            input_ = eval_data[0][idx]
            output_ = eval_data[1][idx]
            
            # Subsample few-shot data for this specific query
            fs_indices = random.sample(range(len(few_shot_data[0])), min(num_few_shot, len(few_shot_data[0])))
            demo_subset = ([few_shot_data[0][i] for i in fs_indices], 
                           [few_shot_data[1][i] for i in fs_indices])

            query = get_query(prompt, eval_template, input_, demos_template, demo_subset)
            queries.append(query)
            ground_truths.append(output_)

    # Batch Generation
    print(f"Evaluating {len(prompts)} prompts on {len(sample_indices)} samples...")
    predictions = model.generate(queries)

    # Scoring (Using utility.py logic)
    # MMLU uses Exact Match (em) usually
    score_fn = utility.get_multi_answer_em 
    
    all_scores = []
    for pred, ans in zip(predictions, ground_truths):
        score = score_fn(pred, ans)
        all_scores.append(score)

    # Reshape and Aggregate
    # shape: (num_prompts, num_samples)
    scores_matrix = np.array(all_scores).reshape(len(prompts), len(sample_indices))
    mean_scores = np.mean(scores_matrix, axis=1)

    # Result: List of (prompt, score)
    results = list(zip(prompts, mean_scores))
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results