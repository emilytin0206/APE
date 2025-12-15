import random
import fire
import os
import requests
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from automatic_prompt_engineer import ape, data
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator

# 定義 MMLU 子集
mmlu_subsets = [
    "high_school_mathematics",
    "high_school_world_history",
    "high_school_physics",
    "professional_law",
    "business_ethics"
]

# ================= 配置區 =================
# 1. Optimizer Model (負責生成 Prompt，建議用強一點的模型)
OPTIMIZER_API_URL = "http://140.113.86.14:11434/api/chat"
OPTIMIZER_MODEL_NAME = "qwen2.5:32b" #這裡可以換成更強的，例如 qwen2.5:32b

# 2. Target Model (實際要測試/部署的模型，負責做題)
TARGET_API_URL = "http://140.113.86.14:11434/api/chat"
TARGET_MODEL_NAME = "qwen2.5:7b"
# ========================================
# ========================================

def format_mmlu_example(example):
    """格式化 MMLU 題目"""
    question = example['question']
    options = example['choices']
    answer_idx = example['answer']
    answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    answer_char = answer_map[answer_idx]
    
    input_str = f"Question: {question}\nOptions:\n"
    for i, opt in enumerate(options):
        input_str += f"{answer_map[i]}. {opt}\n"
    
    return input_str.strip(), [answer_char]

def load_mmlu_data(task, limit=300):
    """載入數據集，最多取 limit 筆"""
    print(f"Loading MMLU subset: {task}...")
    try:
        # 載入測試集 (MMLU 的 test split 比較完整)
        dataset = load_dataset("cais/mmlu", task, split="test")
        
        # 實際取用數量：如果資料夠多就取 limit，不夠就全取
        actual_count = min(limit, len(dataset))
        dataset = dataset.select(range(actual_count))
        
        print(f"  - Loaded {actual_count} examples from {task}")
        
        inputs, outputs = [], []
        for example in dataset:
            inp, out = format_mmlu_example(example)
            inputs.append(inp)
            outputs.append(out)
        return inputs, outputs
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], []

def run_task(task, total_limit=300, train_ratio=0.8):
    """
    依照比例動態切分數據
    :param total_limit: 每個任務最多使用的總題數 (預設 300)
    :param train_ratio: 訓練集佔比 (預設 0.8，即 80% 訓練 / 20% 測試)
    """
    print(f"\n{'='*20} Processing Task: {task} {'='*20}")
    
    # 1. 載入數據 (總量由 total_limit 控制)
    all_inputs, all_outputs = load_mmlu_data(task, limit=total_limit)
    total_len = len(all_inputs)
    
    if total_len == 0:
        print(f"Skipping {task}: No data loaded.")
        return

    # 2. 計算切分點 (關鍵修改：依照實際載入的數量 * 比例)
    n_train = int(total_len * train_ratio)
    n_test = total_len - n_train

    # 防呆機制：如果資料真的太少 (例如只有 2 筆)，強制至少分 1 筆給測試
    if n_test < 1 and total_len > 1:
        n_train = total_len - 1
        n_test = 1
    
    print(f"Dataset Split -> Total: {total_len} | Train: {n_train} | Test: {n_test}")

    if n_train == 0 or n_test == 0:
        print("❌ Data too small to split. Skipping.")
        return

    # 3. 執行切分
    induce_data = (all_inputs[:n_train], all_outputs[:n_train])
    test_data = (all_inputs[n_train:], all_outputs[n_train:])
    
    # 4. 準備 Few-shot 生成用的資料 (從訓練集中再取一小部分)
    # 最多取 20 筆，或是訓練集的一半
    prompt_gen_size = min(int(n_train * 0.5), 20)
    if prompt_gen_size < 1: prompt_gen_size = 1
    
    prompt_gen_data, eval_data = data.create_split(induce_data, prompt_gen_size)
    prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0] for output in prompt_gen_data[1]]

    # 設定模板
    eval_template = "Instruction: [PROMPT]\n\n[INPUT]\nAnswer: [OUTPUT]"
    prompt_gen_template = "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]"
    demos_template = "[INPUT]\nAnswer: [OUTPUT]"
    base_config = '../experiments/configs/instruction_induction.yaml'
    
    # 配置模型
    ollama_optimizer_config = {
        'name': 'Ollama_Forward',
        'api_url': OPTIMIZER_API_URL,
        'batch_size': 1,
        'gpt_config': {
            'model': OPTIMIZER_MODEL_NAME, 
            'temperature': 0.7,
            'max_tokens': 2048,
            'top_p': 0.9
        }
    }
    
    ollama_target_config = {
        'name': 'Ollama_Forward',
        'api_url': TARGET_API_URL,
        'batch_size': 1,
        'gpt_config': {
            'model': TARGET_MODEL_NAME,
            'temperature': 0.0,
            'max_tokens': 1024,
            'top_p': 0.9
        }
    }

    # 確保驗證樣本數不超過實際擁有的數據
    num_internal_eval_samples = min(20, len(eval_data[0]))
    if num_internal_eval_samples == 0:
         eval_data = prompt_gen_data # Fallback

    conf = {
        'generation': {
            'num_subsamples': 3,
            'num_demos': 5,
            'num_prompts_per_subsample': 10,
            'model': ollama_optimizer_config
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': num_internal_eval_samples,
            'model': ollama_target_config
        }
    }

    print(f'Optimizing prompts using {OPTIMIZER_MODEL_NAME}...')
    try:
        res, demo_fn = ape.find_prompts(eval_template=eval_template,
                                        prompt_gen_data=prompt_gen_data,
                                        eval_data=eval_data,
                                        conf=conf,
                                        base_conf=base_config,
                                        few_shot_data=prompt_gen_data,
                                        demos_template=demos_template,
                                        prompt_gen_template=prompt_gen_template)
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return

    prompts, scores = res.sorted()
    if not prompts:
        print("❌ No prompts generated.")
        return

    print('Top Prompts found:')
    for p, s in list(zip(prompts, scores))[:3]:
        print(f'  Score {s:.2f}: {p}')

    # 5. 在測試集上評估 (使用全部測試集資料)
    print(f'Evaluating best prompt on test data ({len(test_data[0])} samples)...')
    test_conf = {
        'generation': conf['generation'],
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': len(test_data[0]), 
            'model': ollama_target_config
        }
    }

    try:
        test_res = ape.evaluate_prompts(prompts=[prompts[0]],
                                        eval_template=eval_template,
                                        eval_data=test_data,
                                        few_shot_data=prompt_gen_data,
                                        demos_template=demos_template,
                                        conf=test_conf,
                                        base_conf=base_config)

        test_score = test_res.sorted()[1][0]
        print(f'Final Test Score for {task}: {test_score}')

        # 存檔
        os.makedirs('experiments/results/instruction_induction', exist_ok=True)
        with open(f'experiments/results/instruction_induction/{task}.txt', 'w', encoding='utf-8') as f:
            f.write(f'Task: {task}\n')
            f.write(f'Data Split: {n_train}/{n_test} (Total {total_len})\n')
            f.write(f'Optimizer: {OPTIMIZER_MODEL_NAME}\n')
            f.write(f'Target: {TARGET_MODEL_NAME}\n')
            f.write(f'Test score: {test_score}\n')
            f.write(f'Best Prompt: {prompts[0]}\n')
            f.write(f'Top 5 Candidates:\n')
            for p, s in list(zip(prompts, scores))[:5]:
                f.write(f'  {s:.2f}: {p}\n')
    except Exception as e:
         print(f"❌ Error during final evaluation: {e}")

def run():
    for task in mmlu_subsets:
        try:
            run_task(task)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"Error in {task}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    fire.Fire(run)