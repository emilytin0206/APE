# ape/llm.py
import requests
from typing import List
from tqdm import tqdm

class LLM:
    def __init__(self, config):
        self.config = config
        self.api_url = config.get('api_url', "http://localhost:11434/api/chat")
        self.model_name = config.get('model', 'qwen2.5:7b')
        self.temperature = config.get('temperature', 0.7)

    def generate(self, prompts: List[str]) -> List[str]:
        results = []
        # Batch size 1 for simple Ollama implementation
        for prompt in tqdm(prompts, desc="LLM Inference"):
            # Remove [APE] token if present (artifact of template)
            clean_prompt = prompt.replace('[APE]', '').strip()
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": clean_prompt}],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 1024 # Max tokens
                }
            }
            
            try:
                resp = requests.post(self.api_url, json=payload, timeout=60)
                resp.raise_for_status()
                content = resp.json().get('message', {}).get('content', '')
                results.append(content)
            except Exception as e:
                print(f"Error: {e}")
                results.append("")
                
        return results