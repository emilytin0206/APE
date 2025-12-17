import ollama
from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def generate_text(self, prompt, n):
        pass


class Ollama_Forward(LLM):
    """
    Wrapper for Ollama (Local LLM).
    """
    def __init__(self, config):
        self.config = config
        
    def generate_text(self, prompt, n=1):
        """
        prompt: 可以是單一字串或字串 list
        n: 每個 prompt 生成幾個回答 (Ollama 通常一次一個，這裡用迴圈模擬)
        """
        if not isinstance(prompt, list):
            prompt = [prompt]
            
        # 讀取參數，如果沒有設定就用預設值
        ollama_config = self.config.get('ollama_config', {})
        model_name = ollama_config.get('model', 'llama3') # 預設用 llama3
        options = {
            'temperature': ollama_config.get('temperature', 0.7),
            'top_p': ollama_config.get('top_p', 0.9),
            # 你可以在這裡加入更多 Ollama 支援的參數
        }

        results = []
        print(f"[Ollama] Generating {len(prompt) * n} completions using {model_name}...")
        
        for p in prompt:
            for _ in range(n):
                try:
                    response = ollama.generate(
                        model=model_name, 
                        prompt=p, 
                        options=options
                    )
                    results.append(response['response'])
                except Exception as e:
                    print(f"Ollama Error: {e}")
                    results.append("") # 錯誤時回傳空字串避免崩潰
                    
        return results

def model_from_config(config):
    model_type = config["name"]
    
    if model_type == "GPT_forward":
        return GPT_Forward(config)
    elif model_type == "GPT_insert":
        return GPT_Insert(config)
    # === 新增這一行 ===
    elif model_type == "Ollama_Forward":
        return Ollama_Forward(config)
    # =================
    
    raise ValueError(f"Unknown model type: {model_type}")

