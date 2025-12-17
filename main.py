# main.py

from ape.template import EvalTemplate, DemosTemplate, GenerationTemplate
from ape.generate import generate_prompts
from ape.evaluator import exec_accuracy_evaluator

# ==========================================
# 1. 定義資料集
# ==========================================
words = ["sane", "direct", "informally", "unpopular"]
antonyms = [["insane"], ["indirect"], ["formally"], ["popular"]]
dataset = (words, antonyms)

# ==========================================
# 2. 雙模型配置 (核心修改)
# ==========================================
ape_config = {
    # --- Optimizer: 負責生成 Prompt ---
    'generation': {
        'num_subsamples': 3,            # 嘗試構建幾次 "逆向工程" 的 Query
        'num_demos': 2,                 # 每次 Query 放幾個範例
        'num_prompts_per_subsample': 2, # 每個 Query 生成幾個 Prompt
        'model': {
            'name': 'Ollama_Forward',
            'ollama_config': {
                'model': 'llama3',      # 使用較強的模型來發想
                'temperature': 0.9,     # 創意度高一點
            }
        }
    },
    
    # --- Scorer: 負責評分驗證 ---
    'evaluation': {
        'num_samples': 10,              # 拿生成的 Prompt 測試幾筆資料
        'num_few_shot': 2,              # 測試時提供幾個範例
        'model': {
            'name': 'Ollama_Forward',
            'batch_size': 1,
            'ollama_config': {
                'model': 'llama3',      # 可以換成 mistral 或同樣用 llama3
                'temperature': 0.0,     # 評分時要穩定，通常設 0
            }
        }
    }
}

# ==========================================
# 3. 準備 Templates
# ==========================================
# 生成用的 Template (逆向工程)
gen_template_str = (
    "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n"
    "[full_DEMO]\n\n"
    "The instruction was to [APE]"
)
# 評估用的 Template (正向測試)
eval_template_str = "Instruction: [PROMPT]\nInput: [INPUT]\n\n[full_DEMO]\n\nOutput: [OUTPUT]"
# 通用範例 Template
demos_template_str = "Input: [INPUT]\nOutput: [OUTPUT]"

gen_template = GenerationTemplate(gen_template_str)
eval_template = EvalTemplate(eval_template_str)
demos_template = DemosTemplate(demos_template_str)

# ==========================================
# 4. 執行流程
# ==========================================

# --- Step 1: Generate (Optimizer) ---
print("\n=== Step 1: Optimizing Prompts ===")
candidates = generate_prompts(
    prompt_gen_template=gen_template,
    demos_template=demos_template,
    prompt_gen_data=dataset,
    config=ape_config['generation']  # <--- 傳入 generation config
)

if not candidates:
    print("No prompts generated.")
    exit()

print(f"Candidates: {candidates}")

# --- Step 2: Evaluate (Scorer) ---
print("\n=== Step 2: Scoring Prompts ===")
results = exec_accuracy_evaluator(
    prompts=candidates,
    eval_template=eval_template,
    eval_data=dataset,
    demos_template=demos_template,
    few_shot_data=dataset,
    config=ape_config['evaluation']  # <--- 傳入 evaluation config
)

# 顯示結果
print("\n=== Final Results ===")
for prompt, score in results:
    print(f"Score: {score:.2f} | Prompt: {prompt}")