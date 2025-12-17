import ollama
import datetime
import os
from abc import ABC, abstractmethod

# 嘗試引用進度條，如果沒有安裝則使用 dummy 函數
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

class LLM(ABC):
    @abstractmethod
    def generate(self, prompt, n):
        pass

class Ollama_Forward(LLM):
    """
    Wrapper for Ollama (Local LLM) with Logging and Progress Bar.
    """
    def __init__(self, config):
        self.config = config
        # 初始化 Client
        host = self.config.get('api_url')
        self.client = ollama.Client(host=host) if host else ollama
        
        # 設定 Log 檔案名稱 (例如: ollama_history.log)
        self.log_file = "ollama_history.log"
        # 啟動時先在 Log 寫一行分隔線，標記新的一次執行
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*20} New Session Started at {datetime.datetime.now()} {'='*20}\n")
        
    def generate(self, prompt, n=1):
        """
        prompt: 可以是單一字串或字串 list
        n: 每個 prompt 生成幾個回答
        """
        if not isinstance(prompt, list):
            prompt = [prompt]
            
        model_name = self.config.get('model', 'llama3')
        
        options = {
            'temperature': self.config.get('temperature', 0.7),
            'top_p': self.config.get('top_p', 0.9),
        }

        results = []
        total_tasks = len(prompt) * n
        print(f"[Ollama] Generating {total_tasks} completions using {model_name}...")
        print(f"[Log] 詳細輸出將記錄於: {os.path.abspath(self.log_file)}")
        
        # 使用 tqdm 顯示進度條
        for p in tqdm(prompt, desc="Gen Progress"):
            for _ in range(n):
                try:
                    # 發送請求
                    response = self.client.generate(
                        model=model_name, 
                        prompt=p, 
                        options=options
                    )
                    res_content = response['response']
                    results.append(res_content)
                    
                    # === [新增功能] 寫入 Log ===
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        f.write(f"[{timestamp}] PROMPT (len={len(p)}):\n{p[:3000]}...\n") # 只印前200字避免太長
                        f.write(f"[{timestamp}] RESPONSE:\n{res_content}\n")
                        f.write("-" * 40 + "\n")
                    # ===========================

                except Exception as e:
                    error_msg = f"Ollama Error: {e}"
                    print(error_msg)
                    results.append("")
                    
                    # 記錄錯誤到 Log
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(f"[ERROR] {error_msg}\n")
                    
        return results

def model_from_config(config):
    model_type = config.get("name")
    if model_type == "Ollama_Forward":
        return Ollama_Forward(config)
    return None