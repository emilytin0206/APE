# ape/evaluator.py
import random
import numpy as np
from typing import List, Tuple, Dict, Any
from . import utility 

# [Fix 1] 修改函式簽章，接收 model 並移除不需要的 template 物件參數
def exec_accuracy_evaluator(
    model, 
    prompts: List[str],
    eval_data: Tuple[List[str], List[List[str]]],
    few_shot_data: Tuple[List[str], List[List[str]]],
    config: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """
    Evaluates prompts using Execution Accuracy (Exact Match).
    Adaptation: Accepts model instance directly and handles string templates.
    """
    
    # 讀取參數
    num_samples = config.get('num_samples', 20)
    num_few_shot = config.get('num_few_shot', 0)
    
    # [Fix 2] 直接從 config 讀取字串模板 (配合 main.py 的設定)
    eval_template = config.get('eval_template', "Instruction: [PROMPT]\n\n[INPUT]\n[OUTPUT]")
    demos_template = config.get('demos_template', "Input: [INPUT]\nOutput: [OUTPUT]")
    
    # 準備數據採樣 (Subsampling)
    inputs, outputs = eval_data
    indices = list(range(len(inputs)))
    # 確保不會 index out of range
    sample_indices = indices[:min(num_samples, len(indices))]
    
    queries = []
    ground_truths = []
    
    print(f"Generating queries for {len(prompts)} prompts over {len(sample_indices)} samples...")

    # 構建查詢
    for prompt in prompts:
        for idx in sample_indices:
            inp = inputs[idx]
            out = outputs[idx] # 這是標準答案 list
            
            # 準備 Few-shot 範例
            full_demo = ""
            if num_few_shot > 0 and few_shot_data:
                # 隨機抽取 demo
                fs_indices = random.sample(range(len(few_shot_data[0])), min(num_few_shot, len(few_shot_data[0])))
                
                demo_strs = []
                for i in fs_indices:
                    # [Fix 3] 使用 replace 處理字串模板
                    d = demos_template.replace('[INPUT]', few_shot_data[0][i])\
                                      .replace('[OUTPUT]', few_shot_data[1][i][0])
                    demo_strs.append(d)
                full_demo = "\n\n".join(demo_strs)

            # [Fix 3] 使用 replace 處理 Eval 模板
            # 邏輯：先把 prompt 放進去，再放 demo，最後放 input
            # 注意：這裡的 replace 順序需依照您的模板結構微調，這裡假設標準 APE 格式
            query = eval_template.replace('[PROMPT]', prompt)\
                                 .replace('[INPUT]', inp)\
                                 .replace('[OUTPUT]', "") # 留空給模型填
            
            # 如果有 demo，通常 eval_template 裡沒有 [full_demo] 佔位符
            # 這裡簡單地將 demo 接在 query 前面 (若模板沒特殊設計)
            if full_demo:
                query = f"{full_demo}\n\n{query}"
            
            queries.append(query)
            ground_truths.append(out)

    # 執行生成 (Batch Generation)
    print(f"Querying LLM (Total {len(queries)} requests)...")
    
    # [Fix 4] 使用 model.generate (配合之前修正的 llm.py)
    # 這裡 queries 是 list，n=1
    predictions = model.generate(queries, n=1)
    
    predictions = [p.strip() for p in predictions]

    # 計算分數 (Exact Match)
    # 使用 utility 中的函數 (請確保 utility.py 存在且有此函數，否則需手寫簡單版)
    try:
        score_fn = utility.get_multi_answer_em
    except AttributeError:
        # Fallback if utility is missing
        def simple_em(pred, target_list):
            return 1.0 if pred in target_list else 0.0
        score_fn = simple_em
    
    all_scores = []
    for pred, ans_list in zip(predictions, ground_truths):
        # ans_list 是一個 list (因為 MMLU 答案可能是多個選項寫法，雖然通常是一個)
        score = score_fn(pred, ans_list)
        all_scores.append(score)

    # 聚合分數
    # reshape: (num_prompts, num_samples)
    if len(all_scores) > 0:
        scores_matrix = np.array(all_scores).reshape(len(prompts), len(sample_indices))
        mean_scores = np.mean(scores_matrix, axis=1)
    else:
        mean_scores = [0.0] * len(prompts)

    # 回傳排序結果
    results = list(zip(prompts, mean_scores))
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results