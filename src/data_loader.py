# src/data_loader.py
import random
from datasets import load_dataset

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self, subset_name):
        """根據 Config 載入並格式化數據"""
        dataset_name = self.config['data']['dataset_name']
        split = self.config['data'].get('split', 'test')
        limit = self.config['data']['limit']

        print(f"Loading {dataset_name} ({subset_name})...")
        try:
            # 這裡保留你原本的邏輯，但改為動態讀取
            dataset = load_dataset(dataset_name, subset_name, split=split)
            actual_count = min(limit, len(dataset))
            dataset = dataset.select(range(actual_count))
            
            inputs, outputs = [], []
            for example in dataset:
                inp, out = self._format_example(example, dataset_name)
                inputs.append(inp)
                outputs.append(out)
                
            return inputs, outputs
        except Exception as e:
            print(f"Error loading dataset {subset_name}: {e}")
            return [], []

    def _format_example(self, example, dataset_name):
        """格式化單個樣本，未來可以在這裡擴充其他數據集的格式"""
        if "mmlu" in dataset_name:
            return self._format_mmlu(example)
        # elif "gsm8k" in dataset_name:
        #     return self._format_gsm8k(example)
        else:
            # 預設格式
            return str(example.get('input', '')), [str(example.get('output', ''))]

    def _format_mmlu(self, example):
        """原本 run_instruction_induction.py 中的邏輯"""
        question = example['question']
        options = example['choices']
        answer_idx = example['answer']
        answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        answer_char = answer_map[answer_idx]
        
        input_str = f"Question: {question}\nOptions:\n"
        for i, opt in enumerate(options):
            input_str += f"{answer_map[i]}. {opt}\n"
        
        return input_str.strip(), [answer_char]

    def split_data(self, inputs, outputs):
        """處理 Train/Test 切分"""
        total_len = len(inputs)
        train_ratio = self.config['data']['train_ratio']
        
        n_train = int(total_len * train_ratio)
        n_test = total_len - n_train
        
        # 防呆機制
        if n_test < 1 and total_len > 1:
            n_train = total_len - 1
            n_test = 1
            
        train_data = (inputs[:n_train], outputs[:n_train])
        test_data = (inputs[n_train:], outputs[n_train:])
        
        return train_data, test_data