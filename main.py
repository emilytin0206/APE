# main.py
import os
import json
import datetime
import random
from ape.llm import LLM
from ape.generate import generate_prompts
from ape.evaluator import exec_accuracy_evaluator
from data.mmlu import load_merged_mmlu_data  # 改用新的合併函數

def save_experiment_results(config, task_name, results):
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
    
    # 定義要合併訓練的 5 個 MMLU 子集
    TRAIN_SUBSETS = [
        "high_school_mathematics",
        "high_school_world_history",
        "high_school_physics",
        "professional_law",
        "business_ethics"
    ]
    
    # 測試集 (通常選其中一個或全部，這裡以 high_school_mathematics 為例進行主要評測)
    TEST_TASK = "high_school_mathematics" 

    conf = {
        'optimizer': {
            'model': 'qwen2.5:32b',
            'api_url': OLLAMA_URL,
            'temperature': 0.9 # 稍微調高溫度以增加生成多樣性
        },
        'target': {
            'model': 'qwen2.5:7b',
            'api_url': OLLAMA_URL,
            'temperature': 0.0
        },
        'generation': {
            # === 官方 3 次迭代設定 ===
            'num_subsamples': 3,             # 迭代 3 次 (3 組不同的 Context)
            'num_prompts_per_subsample': 10, # 每次生成 10 個 (總共 30 個候選)
            'num_demos': 3,
            'prompt_gen_template': "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]",
            'demos_template': "Input: [INPUT]\nOutput: [OUTPUT]"
        },
        'evaluation': {
            'num_samples': 20,
            'num_few_shot': 0,
            'eval_template': "Instruction: [PROMPT]\n\n[INPUT] [OUTPUT]",
            'demos_template': "Input: [INPUT]\nOutput: [OUTPUT]"
        }
    }

    # 1. Load Data (Merging 5 subsets for Training)
    # 限制每個子集只載入 50 筆，避免訓練資料過大 (總共 250 筆)
    inputs, outputs = load_merged_mmlu_data(TRAIN_SUBSETS, limit_per_subset=50)
    
    if not inputs:
        print("Error loading data.")
        return

    # 將所有數據作為訓練數據 (Generation Data)
    train_data = (inputs, outputs)

    # 載入專門的測試數據 (Evaluation Data)
    # 這裡我們重新載入一次 TEST_TASK 作為評估基準
    test_inputs, test_outputs = load_merged_mmlu_data([TEST_TASK], split='test', limit_per_subset=100)
    eval_data = (test_inputs, test_outputs)

    # 2. Generate Prompts (Optimizer)
    optimizer_model = LLM(conf['optimizer'])
    candidates = generate_prompts(optimizer_model, train_data, conf['generation'])
    
    print(f"\nGenerated {len(candidates)} candidates.")
    
    if not candidates:
        print("No candidates generated.")
        return

    # 3. Evaluate Prompts (Target)
    target_model = LLM(conf['target'])
    scored_results = exec_accuracy_evaluator(
        target_model, 
        candidates, 
        eval_data,     # 在測試集上評估
        train_data,    # 如果需要 Few-shot，從訓練集取樣
        conf['evaluation']
    )

    # 4. Show & Save Results
    print("\n=== Final Leaderboard ===")
    for rank, (prompt, score) in enumerate(scored_results[:5]):
        print(f"Rank {rank+1} | Score: {score:.2f} | Prompt: {prompt}")

    save_experiment_results(conf, "merged_5_tasks", scored_results)

if __name__ == "__main__":
    main()