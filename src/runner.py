# src/runner.py
import os
import random
import yaml
import json
from src.evaluator import exec_accuracy_evaluator
from automatic_prompt_engineer import ape, data
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
from src.data_loader import DataLoader

class APEExperimentRunner:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.data_loader = DataLoader(self.config)

    def _get_model_config(self, role):
        """產生 APE 需要的模型配置字典"""
        api_conf = self.config['api']
        if role == 'optimizer':
            return {
                'name': 'Ollama_Forward',
                'api_url': api_conf['optimizer_url'],
                'batch_size': 1,
                'gpt_config': {
                    'model': api_conf['optimizer_model'], 
                    'temperature': 0.7,
                    'max_tokens': 2048,
                    'top_p': 0.9
                }
            }
        elif role == 'target':
            return {
                'name': 'Ollama_Forward',
                'api_url': api_conf['target_url'],
                'batch_size': 1,
                'gpt_config': {
                    'model': api_conf['target_model'],
                    'temperature': 0.0,
                    'max_tokens': 1024,
                    'top_p': 0.9
                }
            }

    def run_single_task(self, task_name):
        print(f"\n{'='*20} Processing Task: {task_name} {'='*20}")
        
        # 1. 載入與切分數據
        inputs, outputs = self.data_loader.load_data(task_name)
        if not inputs:
            return
            
        train_data, test_data = self.data_loader.split_data(inputs, outputs)
        
        # 2. 準備 Few-shot 資料
        prompt_gen_size = min(int(len(train_data[0]) * 0.5), self.config['experiment']['prompt_gen_size'])
        if prompt_gen_size < 1: prompt_gen_size = 1
        
        prompt_gen_data, eval_data = data.create_split(train_data, prompt_gen_size)
        # 隨機抽取一個答案作為示範
        prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0] for output in prompt_gen_data[1]]

        # 3. 設定 APE 參數
        # 這裡引用原本 run_instruction_induction.py 中的 template 設定
        eval_template = "Instruction: [PROMPT]\n\n[INPUT]\nAnswer: [OUTPUT]"
        prompt_gen_template = "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]"
        demos_template = "[INPUT]\nAnswer: [OUTPUT]"
        
        base_ape_config = 'experiments/configs/instruction_induction.yaml' # 確保路徑正確或改為絕對路徑
        
        conf = {
            'generation': {
                'num_subsamples': 3,
                'num_demos': 5,
                'num_prompts_per_subsample': 10,
                'model': self._get_model_config('optimizer')
            },
            'evaluation': {
                'method': exec_accuracy_evaluator,
                'task': task_name,
                'num_samples': min(20, len(eval_data[0])),
                'model': self._get_model_config('target')
            }
        }

        # 4. 執行優化 (Find Prompts)
        print(f"Optimizing prompts...")
        res, demo_fn = ape.find_prompts(
            eval_template=eval_template,
            prompt_gen_data=prompt_gen_data,
            eval_data=eval_data,
            conf=conf,
            base_conf=base_ape_config,
            few_shot_data=prompt_gen_data,
            demos_template=demos_template,
            prompt_gen_template=prompt_gen_template
        )

        prompts, scores = res.sorted()
        if not prompts:
            print("No prompts generated.")
            return

        best_prompt = prompts[0]
        print(f"Best Prompt: {best_prompt}")

        # 5. 最終測試 (Evaluate on Test Set)
        print(f"Evaluating on test set ({len(test_data[0])} samples)...")
        test_conf = {
            'generation': conf['generation'],
            'evaluation': {
                'method': exec_accuracy_evaluator,
                'task': task_name,
                'num_samples': len(test_data[0]),
                'model': self._get_model_config('target')
            }
        }
        
        test_res = ape.evaluate_prompts(
            prompts=[best_prompt],
            eval_template=eval_template,
            eval_data=test_data,
            few_shot_data=prompt_gen_data,
            demos_template=demos_template,
            conf=test_conf,
            base_conf=base_ape_config
        )
        
        test_score = test_res.sorted()[0][1] # (prompt, score) tuple
        print(f"Final Test Score: {test_score}")
        
        self._save_results(task_name, best_prompt, test_score, prompts, scores)

    def _save_results(self, task, best_prompt, score, prompts, scores):
        """存檔為 JSON 格式，比原本的 TXT 更易於程式讀取"""
        out_dir = self.config['experiment']['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        
        result = {
            "task": task,
            "optimizer_model": self.config['api']['optimizer_model'],
            "target_model": self.config['api']['target_model'],
            "test_score": score,
            "best_prompt": best_prompt,
            "candidates": [{"prompt": p, "score": s} for p, s in zip(prompts, scores)[:5]]
        }
        
        with open(os.path.join(out_dir, f"{task}.json"), 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    def run_all(self):
        for task in self.config['data']['subsets']:
            try:
                self.run_single_task(task)
            except Exception as e:
                print(f"Failed task {task}: {e}")