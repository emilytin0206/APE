# main.py
import os
import json
import datetime
import random
from ape.llm import LLM
from ape.generate import generate_prompts
from ape.evaluator import exec_accuracy_evaluator
from data.mmlu import load_merged_mmlu_data

def save_experiment_results(config, task_name, results):
    # (保持原樣，省略以節省篇幅)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_model_safe = config['target']['model'].replace(':', '-')
    optimizer_model_safe = config['optimizer']['model'].replace(':', '-')
    
    dir_name = f"{target_model_safe}_{optimizer_model_safe}_{task_name}_{timestamp}"
    base_dir = os.path.join("experiments", "results")
    save_dir = os.path.join(base_dir, dir_name)
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Saving] 儲存實驗結果至: {save_dir}")

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    output_data = []
    for rank, (prompt, score) in enumerate(results):
        output_data.append({"rank": rank + 1, "score": score, "prompt": prompt})

    with open(os.path.join(save_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    if output_data:
        best_result = output_data[0]
        with open(os.path.join(save_dir, "best_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(f"Experiment: {dir_name}\nTask: {task_name}\nBest Score: {best_result['score']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Best Prompt:\n{best_result['prompt']}\n")
            f.write("-" * 50 + "\n")
    print("[Saving] 儲存完成！")

def main():
    # --- Configuration ---
    OLLAMA_URL = "http://140.113.86.14:11434/api/chat"
    
    # 1. 定義所有要參與實驗的子集
    ALL_SUBSETS = [
        "high_school_mathematics",
        "high_school_world_history",
        "high_school_physics",
        "professional_law",
        "business_ethics"
    ]
    
    # 定義實驗標籤名稱
    TASK_LABEL = f"merged_{len(ALL_SUBSETS)}_subsets"

    conf = {
        'optimizer': {
            'model': 'qwen2.5:32b',
            'api_url': OLLAMA_URL,
            'temperature': 0.7 # 稍微調高溫度以增加生成多樣性
        },
        'target': {
            'model': 'qwen2.5:7b',
            'api_url': OLLAMA_URL,
            'temperature': 0.0
        },
        'generation': {
            # === 官方 3 次迭代設定 ===
            'num_subsamples': 5,            
            'num_prompts_per_subsample': 50, # 每次生成 n 個 
            'num_demos': 5,
            'prompt_gen_template': "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]",
            'demos_template': "Input: [INPUT]\nOutput: [OUTPUT]"
        },
        'evaluation': {
            'num_samples': 50,
            'num_few_shot': 0,
            'eval_template': "Instruction: [PROMPT]\n\n[INPUT] [OUTPUT]",
            'demos_template': "Input: [INPUT]\nOutput: [OUTPUT]"
        }
    }
# --- Step 1: Data Loading & Pooling ---
    print(f"\n[Data] Loading and merging {len(ALL_SUBSETS)} subsets...")
    
    # 這裡我們一次讀取所有子集，限制每個子集讀取量 (例如各 100 筆，總共 500 筆)
    # 這樣可以確保數據集不會過大，但包含所有領域
    raw_inputs, raw_outputs = load_merged_mmlu_data(ALL_SUBSETS, split='test', limit_per_subset=100)
    
    if not raw_inputs:
        print("[Error] No data loaded.")
        return

    # --- Step 2: Shuffling & Strict Splitting ---
    print("[Data] Shuffling and splitting data...")
    
    # 將 input/output 配對後一起洗牌，確保對應關係不變
    paired_data = list(zip(raw_inputs, raw_outputs))
    random.seed(42) # 固定種子，確保實驗可重現
    random.shuffle(paired_data)
    
    # 解開配對
    shuffled_inputs, shuffled_outputs = zip(*paired_data)
    shuffled_inputs = list(shuffled_inputs)
    shuffled_outputs = list(shuffled_outputs)

    # 定義切分點 (Split Point)
    # 例如：拿前 50 筆做 APE 的訓練 (生成 Prompt)，剩下的做測試
    TRAIN_SIZE = 50 
    
    # Strict Split: 
    # train_data 用於 generate_prompts (含 demos)
    train_data = (shuffled_inputs[:TRAIN_SIZE], shuffled_outputs[:TRAIN_SIZE])
    
    # eval_data 用於 evaluator (計算分數)
    eval_data = (shuffled_inputs[TRAIN_SIZE:], shuffled_outputs[TRAIN_SIZE:])

    print(f"  - Total Data Pool: {len(shuffled_inputs)}")
    print(f"  - Train Set (Generation Demos): {len(train_data[0])} samples (Indices 0-{TRAIN_SIZE})")
    print(f"  - Test Set (Evaluation): {len(eval_data[0])} samples (Indices {TRAIN_SIZE}-end)")
    
    # --- Step 3: Generate Prompts ---
    print("\n[APE] Step 1: Generating Prompts...")
    optimizer_model = LLM(conf['optimizer'])
    
    # 注意：這裡只傳入 train_data。generate_prompts 內部的 random.sample 
    # 只會從這 50 筆中挑選，絕對不會碰到 eval_data。
    candidates = generate_prompts(optimizer_model, train_data, conf['generation'])
    
    print(f"Generated {len(candidates)} candidates.")
    if not candidates:
        return

    # --- Step 4: Evaluate Prompts ---
    print("\n[APE] Step 2: Evaluating Prompts...")
    target_model = LLM(conf['target'])
    
    # 注意：這裡傳入 eval_data 作為測試題庫。
    # 如果 evaluator 需要 few-shot examples (目前 num_few_shot=0)，
    # 我們傳入 train_data 作為 "few_shot_data" 的來源，確保不在 Test Set 裡面偷看答案。
    scored_results = exec_accuracy_evaluator(
        model=target_model, 
        prompts=candidates, 
        eval_data=eval_data,     # 用 Test Set 考試
        few_shot_data=train_data,# 用 Train Set 當作範例 (如果有開 Few-shot)
        config=conf['evaluation']
    )

    # --- Step 5: Save ---
    print("\n=== Final Leaderboard ===")
    for rank, (prompt, score) in enumerate(scored_results[:5]):
        print(f"Rank {rank+1} | Score: {score:.2f} | Prompt: {prompt}")

    save_experiment_results(conf, TASK_LABEL, scored_results)

if __name__ == "__main__":
    main()