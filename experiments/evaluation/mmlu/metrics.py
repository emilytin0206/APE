# 檔案位置: experiments/evaluation/mmlu/metrics.py
import re
import string

def _extract_bracketed_choice_from_string(s):
    """
    從字串中提取括號選項，例如從 "The answer is (A)." 提取 "A"。
    邏輯：尋找 (A), (B), (C), (D) 模式。
    如果恰好只發現一個選項，則回傳該選項；否則回傳 None (或原字串處理)。
    """
    # 匹配 (A), (B), (C), (D) 或 (a), (b), (c), (d)
    # 使用不區分大小寫的匹配
    matches = re.findall(r'\(([a-d])\)', s, re.IGNORECASE)
    
    # 如果找到唯一的一個選項
    if len(set(matches)) == 1:
        return matches[0].upper()
    
    # 如果找到多個不同的選項，或者沒找到，嘗試尋找沒有括號但在句尾的 A/B/C/D
    # 例如 "Answer: A"
    # 這邊保留擴充空間，目前先依據您的描述僅處理括號
    return None

def get_normalized_prediction(prediction):
    """
    對模型預測結果進行標準化清理與提取。
    """
    # 1. 移除常見引導詞 (Case-insensitive replace)
    prefixes = [
        "The answer is", 
        "The correct answer is",
        "Answer:", 
        "Step-by-Step Answer:",
        "Therefore, the answer is"
    ]
    
    clean_pred = prediction.strip()
    
    for prefix in prefixes:
        # 使用正則表達式進行不區分大小寫的替換
        pattern = re.compile(re.escape(prefix), re.IGNORECASE)
        clean_pred = pattern.sub("", clean_pred)

    # 2. 嘗試提取括號選項 (如 (A))
    bracketed_choice = _extract_bracketed_choice_from_string(clean_pred)
    if bracketed_choice:
        return bracketed_choice

    # 3. 若無括號選項，進行一般標準化清理
    # 轉小寫
    clean_pred = clean_pred.lower()
    # 移除標點符號 (保留數字與字母)
    clean_pred = clean_pred.translate(str.maketrans('', '', string.punctuation))
    # 移除多餘空白
    clean_pred = " ".join(clean_pred.split())
    
    # 4. 最後嘗試捕捉單一字母答案 (例如模型只回傳 "a" 或 "b")
    # 如果清理後只剩下一個字母 a-d，就當作答案
    if len(clean_pred) == 1 and clean_pred in ['a', 'b', 'c', 'd']:
        return clean_pred.upper()
        
    return clean_pred