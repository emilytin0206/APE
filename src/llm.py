# src/llm.py
import os
import time
import requests
import json
from abc import ABC, abstractmethod
from tqdm import tqdm

class LLM(ABC):
    @abstractmethod
    def generate_text(self, prompt, n):
        pass

def model_from_config(config):
    """Factory function to create LLM instance from config"""
    model_type = config.get("name")
    if model_type == "Ollama_Forward":
        return Ollama_Forward(config)
    elif model_type == "GPT_Forward":
        return GPT_Forward(config)
    # 可以在此擴充其他模型
    raise ValueError(f"Unknown model type: {model_type}")

class Ollama_Forward(LLM):
    """針對 Ollama API 的封裝"""
    def __init__(self, config):
        self.config = config
        self.api_url = config.get('api_url', "http://localhost:11434/api/chat")
        self.gpt_config = config.get('gpt_config', {})
        self.disable_tqdm = config.get('disable_tqdm', True)

    def generate_text(self, prompts, n):
        """
        prompts: list of strings
        n: number of generations per prompt
        """
        if not isinstance(prompts, list):
            prompts = [prompts]

        results = []
        batch_size = self.config.get('batch_size', 1)
        
        # 顯示進度條
        iterator = tqdm(prompts, disable=self.disable_tqdm, desc="Generating (Ollama)")
        
        for prompt in iterator:
            # 去除 APE 標記，避免模型困惑
            clean_prompt = prompt.replace('[APE]', '').strip()
            
            prompt_responses = []
            for _ in range(n):
                response = self._call_api(clean_prompt)
                prompt_responses.append(response)
            
            # Ollama 通常回傳單一字串，如果 n>1，我們將結果展平或依需求處理
            # 這裡為了相容性，我們將 n 個結果都加入 results
            results.extend(prompt_responses)
            
        return results

    def _call_api(self, prompt):
        payload = {
            "model": self.gpt_config.get('model', 'llama3'),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.gpt_config.get('temperature', 0.7),
                "num_predict": self.gpt_config.get('max_tokens', 256),
                "top_p": self.gpt_config.get('top_p', 0.9),
            }
        }

        # 重試機制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                
                if 'message' in data:
                    return data['message']['content']
                elif 'response' in data:
                    return data['response']
                else:
                    return ""
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error calling Ollama: {e}")
                    return ""
                time.sleep(2)
        return ""

class GPT_Forward(LLM):
    """保留 GPT 支援作為備用"""
    def __init__(self, config):
        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError("Use `pip install openai` to use GPT models.")
        self.config = config

    def generate_text(self, prompts, n):
        # 簡化版的 GPT 呼叫邏輯，實際使用需設定 API Key
        if not isinstance(prompts, list):
            prompts = [prompts]
            
        gpt_config = self.config.get('gpt_config', {})
        results = []
        
        for prompt in prompts:
            clean_prompt = prompt.replace('[APE]', '').strip()
            try:
                response = self.openai.Completion.create(
                    prompt=clean_prompt,
                    n=n,
                    **gpt_config
                )
                results.extend([choice['text'] for choice in response['choices']])
            except Exception as e:
                print(f"GPT Error: {e}")
                results.extend([""] * n)
        return results