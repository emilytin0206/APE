# ape/generate.py
import random
from typing import List, Tuple, Dict, Any
from . import llm  # 使用你更新過支援 Ollama 的 llm
from .template import GenerationTemplate, DemosTemplate # 假設你有這些 class

def get_query(prompt_gen_template, demos_template, subsampled_data):
    """
    組合生成 Prompt 的 Query
    """
    inputs, outputs = subsampled_data
    # 填充 Few-shot 範例
    demos = demos_template.fill(subsampled_data)
    # 填充生成指令 (輸入/輸出/範例)
    # 這裡的 input/output 是指用來讓模型 "逆向工程" 的那一對資料
    return prompt_gen_template.fill(input=inputs[0], output=outputs[0], full_demo=demos)

def generate_prompts(
    prompt_gen_template: Any,
    demos_template: Any,
    prompt_gen_data: Tuple[List[str], List[List[str]]],
    config: Dict[str, Any]
) -> List[str]:
    """
    使用 Config 設定的 Optimizer 模型來生成候選 Prompts
    """
    
    # 1. 從 Config 實例化 Optimizer 模型
    print(f"Loading Optimizer model: {config['model']['name']}...")
    model = llm.model_from_config(config['model'])

    # 讀取參數
    num_subsamples = config.get('num_subsamples', 5) # 要生成幾次 Query
    num_demos = config.get('num_demos', 3)           # 每個 Query 放幾個範例
    num_prompts_per_subsample = config.get('num_prompts_per_subsample', 1) # 每個 Query 生成幾個結果

    queries = []
    inputs, outputs = prompt_gen_data

    # 2. 構建 Queries
    for _ in range(num_subsamples):
        # 隨機採樣數據
        if len(inputs) > 0:
            indices = random.sample(range(len(inputs)), min(num_demos + 1, len(inputs)))
            # 取第一筆作為 Target (要模型猜指令的對象)，剩下的作為 Context Demos
            target_idx = indices[0]
            demo_indices = indices[1:]
            
            sub_inputs = [inputs[i] for i in demo_indices]
            sub_outputs = [outputs[i] for i in demo_indices]
            
            target_input = inputs[target_idx]
            target_output = outputs[target_idx][0] if isinstance(outputs[target_idx], list) else outputs[target_idx]
            
            # 使用 Template 組合
            # 注意：這裡邏輯需配合你的 Template 實作，這邊模擬官方邏輯
            # 官方邏輯是傳入 tuple (inputs, outputs) 給 demos_template
            demos_str = demos_template.fill((sub_inputs, sub_outputs))
            
            query = prompt_gen_template.fill(
                input=target_input,
                output=target_output,
                full_demo=demos_str
            )
            queries.append(query)

    # 3. 執行生成 (Optimizer)
    print(f"Generating prompts using Optimizer...")
    # 注意：這裡使用 generate_text，n 控制生成的數量
    prompts = model.generate_text(queries, n=num_prompts_per_subsample)

    # 4. 去重與清理
    unique_prompts = list(set([p.strip() for p in prompts if p.strip()]))
    print(f"Generated {len(unique_prompts)} unique prompts.")
    
    return unique_prompts