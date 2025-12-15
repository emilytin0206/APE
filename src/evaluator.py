# src/evaluator.py
import numpy as np
import random
from src import llm, metrics

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.task = config.get('task', 'unknown')
        self.metric_name = metrics.TASK_TO_METRIC.get(self.task, metrics.DEFAULT_METRIC)

    def evaluate(self, prompts, eval_data, few_shot_data, demos_template, eval_template):
        """
        執行評估的主函數
        eval_data: (inputs, outputs) tuple
        few_shot_data: (inputs, outputs) tuple
        """
        # 載入模型
        model = llm.model_from_config(self.config['model'])
        
        queries = []
        answers_list = []
        
        # 準備查詢
        num_samples = self.config.get('num_samples', 10)
        num_few_shot = self.config.get('num_few_shot', 5)
        
        # 為了效率，先隨機採樣一次數據 (原本邏輯是每個 prompt 重新採樣，這會導致比較不公平，固定採樣較好)
        total_eval_indices = list(range(len(eval_data[0])))
        sampled_indices = random.sample(total_eval_indices, min(num_samples, len(total_eval_indices)))
        
        sampled_inputs = [eval_data[0][i] for i in sampled_indices]
        sampled_outputs = [eval_data[1][i] for i in sampled_indices]

        # 每個 Prompt 都跑一次這些樣本
        for prompt in prompts:
            for inp, outp in zip(sampled_inputs, sampled_outputs):
                # 建構 Few-shot 範例
                demo_text = self._build_demos(few_shot_data, demos_template, num_few_shot)
                
                # 填充最終查詢字串
                query = eval_template.replace('[PROMPT]', prompt) \
                                     .replace('[INPUT]', inp) \
                                     .replace('[full_DEMO]', demo_text) \
                                     .replace('[OUTPUT]', '') # 留空給模型填
                
                queries.append(query)
                answers_list.append(outp)

        # 批量生成
        print(f"Evaluating {len(prompts)} prompts on {len(sampled_inputs)} samples (Metric: {self.metric_name})...")
        predictions = model.generate_text(queries, 1)
        
        # 計算分數
        all_scores = []
        for pred, ans in zip(predictions, answers_list):
            score = metrics.score_function(self.metric_name, pred, ans)
            all_scores.append(score)
            
        # Reshape: (num_prompts, num_samples)
        scores_matrix = np.array(all_scores).reshape(len(prompts), len(sampled_inputs))
        
        # 計算每個 Prompt 的平均分
        mean_scores = np.mean(scores_matrix, axis=1)
        
        # 回傳排序後的結果 [(prompt, score), ...]
        result_pairs = list(zip(prompts, mean_scores))
        result_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return result_pairs

    def _build_demos(self, few_shot_data, template, k):
        """構建 Few-shot 示範文本"""
        indices = random.sample(range(len(few_shot_data[0])), min(k, len(few_shot_data[0])))
        demos = []
        for i in indices:
            inp = few_shot_data[0][i]
            # 隨機選一個正確答案做示範
            out = random.choice(few_shot_data[1][i]) 
            demo = template.replace('[INPUT]', inp).replace('[OUTPUT]', out)
            demos.append(demo)
        return "\n\n".join(demos)

# 為了相容 Runner 的介面，提供一個 Wrapper function
def exec_accuracy_evaluator(prompts, eval_template, eval_data, demos_template, few_shot_data, config):
    evaluator = Evaluator(config)
    # 注意：這裡回傳格式需稍微適配原本 APE 的期待，或者我們直接改 Runner 用新的介面
    # 這裡我們回傳一個簡單的物件模擬原本的 EvaluationResult
    results = evaluator.evaluate(prompts, eval_data, few_shot_data, demos_template, eval_template)
    
    class SimpleResult:
        def sorted(self):
            prompts, scores = zip(*results)
            return prompts, scores
            
    return SimpleResult()