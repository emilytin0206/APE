# 檔案位置: experiments/run_mmlu.py
import random
import fire
from automatic_prompt_engineer import ape, data

# 匯入我們剛剛寫的模組
from experiments.data.mmlu.load_data import load_data
from experiments.evaluation.mmlu.mmlu_eval import mmlu_accuracy_evaluator

def run(task_name='abstract_algebra'):
    print(f"Running APE on MMLU task: {task_name}")
    
    # 1. 載入資料
    # 使用 'test' 作為評估集，'dev' 作為生成提示用的資料集 (避免污染)
    induce_data = load_data(task_name, split='dev') 
    eval_data = load_data(task_name, split='test')

    # 從 induce_data 中切分出用於「生成 Prompt」的資料
    prompt_gen_size = min(20, len(induce_data[0]))
    prompt_gen_data, _ = data.create_split(induce_data, prompt_gen_size)

    # 2. 定義 Template (這是 APE 最關鍵的部分)
    
    # [PROMPT]: APE 生成的指令位置 (例如 "Please choose the correct answer.")
    # [INPUT]: 題目 + 選項
    # [OUTPUT]: 標準答案 (例如 "A")
    eval_template = "Instruction: [PROMPT]\n\n[full_DEMO]\n\nQuestion:\n[INPUT]\nAnswer: [OUTPUT]"
    
    # 用於生成 Prompt 的 Template (Reverse Generation)
    # 意思是：給定題目和答案，請模型反推「指令」是什麼
    prompt_gen_template = "I gave a student an instruction. Based on the instruction they answered the following multiple-choice questions:\n\n[full_DEMO]\n\nThe instruction was to [APE]"
    
    demos_template = "Question:\n[INPUT]\nAnswer: [OUTPUT]"

    # 3. 設定參數
    base_config = 'experiments/configs/instruction_induction.yaml' # 可以沿用現有的 config
    conf = {
        'generation': {
            'num_subsamples': 3,          # 生成 Prompt 時採樣幾組資料
            'num_demos': 3,               # 每一組包含幾個範例 (Few-shot)
            'num_prompts_per_subsample': 10, # 每組生成幾個候選 Prompt
            'model': {
                'gpt_config': {
                    'model': 'gpt-3.5-turbo-instruct' # 或其他您想用的模型
                }
            }
        },
        'evaluation': {
            'method': mmlu_accuracy_evaluator, # 使用我們自定義的評估器
            'task': task_name,
            'num_few_shot': 0,             # 評估時是否要 Zero-shot (通常 MMLU 測 0-shot 或 5-shot)
            'num_samples': min(50, len(eval_data[0])), # 為了省錢，先測 50 題
            'model': {
                'gpt_config': {
                     'model': 'gpt-3.5-turbo-instruct',
                     'max_tokens': 5  # 只需要輸出 A/B/C/D，不用太長
                }
            }
        }
    }

    # 4. 執行 APE
    res, demo_fn = ape.find_prompts(
        eval_template=eval_template,
        prompt_gen_data=prompt_gen_data,
        eval_data=eval_data,
        conf=conf,
        base_conf=base_config,
        few_shot_data=induce_data, # 用於填充 [full_DEMO]
        demos_template=demos_template,
        prompt_gen_template=prompt_gen_template
    )

    print('Finished finding prompts.')
    prompts, scores = res.sorted()
    
    print('Top Prompts:')
    for prompt, score in list(zip(prompts, scores))[:5]:
        print(f'  Score {score:.2f}: {prompt}')

    # 儲存結果
    with open(f'experiments/results_mmlu_{task_name}.txt', 'w') as f:
        f.write(f'Best Prompt: {prompts[0]}\n')
        f.write(f'Score: {scores[0]}\n')

if __name__ == '__main__':
    fire.Fire(run)