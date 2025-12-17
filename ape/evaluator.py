# ape/evaluator.py
import random
import numpy as np
from typing import List, Tuple, Dict, Any
from . import llm, utility  # 假設 utility 在同目錄

def exec_accuracy_evaluator(
    prompts: List[str],
    eval_template: Any,  # 傳入 EvalTemplate 物件
    eval_data: Tuple[List[str], List[List[str]]],
    demos_template: Any, # 傳入 DemosTemplate 物件
    few_shot_data: Tuple[List[str], List[List[str]]],
    config: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """
    仿照官方風格的評估器，但使用 Execution Accuracy (Exact Match) 作為評分標準。
    """
    
    # 1. 從 Config 實例化模型 (使用官方 llm.py 的工廠模式)
    # Config 結構需包含 ['model']
    model = llm.model_from_config(config['model'])
    
    # 讀取參數
    num_samples = config.get('num_samples', 20)
    num_few_shot = config.get('num_few_shot', 0)
    
    # 2. 準備數據採樣 (Subsampling)
    inputs, outputs = eval_data
    indices = list(range(len(inputs)))
    # 簡單隨機採樣，或者你可以保留原本固定的 indices[:num_samples]
    sample_indices = indices[:min(num_samples, len(indices))]
    
    queries = []
    ground_truths = []
    
    print(f"Generating queries for {len(prompts)} prompts over {len(sample_indices)} samples...")

    # 3. 構建查詢 (使用 Template 物件)
    for prompt in prompts:
        for idx in sample_indices:
            inp = inputs[idx]
            out = outputs[idx]
            
            # 準備 Few-shot 範例
            if num_few_shot > 0 and few_shot_data:
                # 隨機抽取 demo
                fs_indices = random.sample(range(len(few_shot_data[0])), min(num_few_shot, len(few_shot_data[0])))
                fs_inputs = [few_shot_data[0][i] for i in fs_indices]
                fs_outputs = [few_shot_data[1][i] for i in fs_indices]
                
                # 使用 DemosTemplate 填空
                full_demo = demos_template.fill((fs_inputs, fs_outputs))
            else:
                full_demo = ""

            # 使用 EvalTemplate 填空
            # 注意：這裡 output 留空，讓模型生成
            query = eval_template.fill(prompt=prompt, input=inp, full_demo=full_demo, output="")
            
            queries.append(query)
            ground_truths.append(out)

    # 4. 執行生成 (Batch Generation)
    # 注意：這裡呼叫的是官方 llm.py 的 generate_text，需要 n=1
    print(f"Querying LLM...")
    predictions = model.generate_text(queries, n=1)
    
    # 官方 generate_text 有時回傳 list of lists (如果 n>1)，或是 list of strings
    # 這裡做個簡單處理確保格式統一
    predictions = [p.strip() for p in predictions]

    # 5. 計算分數 (Exact Match)
    score_fn = utility.get_multi_answer_em # 沿用你原本的 utility
    
    all_scores = []
    for pred, ans in zip(predictions, ground_truths):
        score = score_fn(pred, ans)
        all_scores.append(score)

    # 6. 聚合分數
    # reshape: (num_prompts, num_samples)
    scores_matrix = np.array(all_scores).reshape(len(prompts), len(sample_indices))
    mean_scores = np.mean(scores_matrix, axis=1)

    # 回傳排序結果
    results = list(zip(prompts, mean_scores))
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results