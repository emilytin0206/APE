# 檔案位置: experiments/evaluation/mmlu/mmlu_eval.py
import numpy as np
from automatic_prompt_engineer import llm, data, evaluate

# 引入剛剛建立的 metrics
from experiments.evaluation.mmlu.metrics import get_normalized_prediction

def get_query(prompt, eval_template, input_, output_, demo_data, demos_template):
    demos = demos_template.fill(demo_data)
    query = eval_template.fill(prompt=prompt,
                               input=input_,
                               output='',
                               full_demo=demos)
    return query

def mmlu_accuracy_evaluator(prompts, eval_template, eval_data, demos_template, few_shot_data, config):
    queries = []
    ground_truths = []
    
    # 準備 Query
    for prompt in prompts:
        subsampled_data = data.subsample_data(eval_data, config['num_samples'])
        for input_, output_ in zip(*subsampled_data):
            demo_data = data.subsample_data(few_shot_data, config['num_few_shot'])
            query = get_query(prompt, eval_template, input_, output_, demo_data, demos_template)
            
            queries.append(query)
            ground_truths.append(output_)

    # 執行 LLM 生成
    # 注意：針對 MMLU，建議稍微增加 max_tokens 以容許模型輸出 "(A)" 或簡短解釋
    model = llm.model_from_config(config['model'])
    model_outputs = model.generate_text(queries, 1)

    scores = []
    for prediction, truth in zip(model_outputs, ground_truths):
        # --- 使用新的標準化函數 ---
        pred_clean = get_normalized_prediction(prediction)
        truth_clean = truth.strip().upper() # 確保標準答案也是乾淨的 (A, B, C, D)
        
        # 比對 (Exact Match)
        # 因為 get_normalized_prediction 會盡量回傳大寫字母 A/B/C/D
        # 所以這裡直接比對即可
        score = 1.0 if pred_clean == truth_clean else 0.0
        scores.append(score)

    scores = np.array(scores).reshape(len(prompts), config['num_samples'])
    return MMLUEvaluationResult(prompts, scores)

class MMLUEvaluationResult(evaluate.EvaluationResult):
    def __init__(self, prompts, scores):
        self.prompts = prompts
        self.scores = scores

    def sorted(self, method='mean'):
        if method == 'mean':
            agg_scores = [np.mean(s) for s in self.scores]
        else:
            agg_scores = [np.mean(s) for s in self.scores]
            
        sorted_items = sorted(zip(agg_scores, self.prompts), key=lambda x: x[0], reverse=True)
        return [p for _, p in sorted_items], [s for s, _ in sorted_items]