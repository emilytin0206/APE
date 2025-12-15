# main.py
from ape.llm import LLM
from ape.generate import generate_prompts
from ape.evaluator import exec_accuracy_evaluator
from data.mmlu import load_mmlu_data

def main():
    # --- Configuration ---
    TASK = "high_school_mathematics"
    OLLAMA_URL = "http://140.113.86.14:11434/api/chat"
    
    conf = {
        'optimizer': {
            'model': 'qwen2.5:32b',
            'api_url': OLLAMA_URL,
            'temperature': 0.7
        },
        'target': {
            'model': 'qwen2.5:7b',
            'api_url': OLLAMA_URL,
            'temperature': 0.0 # Greedy decoding for evaluation
        },
        'generation': {
            'num_candidates': 10,  # Generate 10 prompts
            'num_demos': 3,
            'prompt_gen_template': "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]",
            'demos_template': "Input: [INPUT]\nOutput: [OUTPUT]"
        },
        'evaluation': {
            'num_samples': 20,     # Evaluate on 20 samples per prompt
            'num_few_shot': 0,     # Zero-shot evaluation for MMLU
            'eval_template': "Instruction: [PROMPT]\n\n[INPUT] [OUTPUT]", # Input contains "Answer:" already
            'demos_template': "Input: [INPUT]\nOutput: [OUTPUT]"
        }
    }

    # 1. Load Data
    inputs, outputs = load_mmlu_data(TASK, limit=100)
    
    # Split Data (Simple 50/50 split for demo)
    mid = len(inputs) // 2
    train_data = (inputs[:mid], outputs[:mid])
    eval_data = (inputs[mid:], outputs[mid:])

    # 2. Generate Prompts (Optimizer)
    optimizer_model = LLM(conf['optimizer'])
    candidates = generate_prompts(optimizer_model, train_data, conf['generation'])
    
    print(f"\nGenerated {len(candidates)} candidates.")
    for i, c in enumerate(candidates):
        print(f"  {i}: {c}")

    if not candidates:
        print("No candidates generated.")
        return

    # 3. Evaluate Prompts (Target)
    target_model = LLM(conf['target'])
    scored_results = exec_accuracy_evaluator(
        target_model, 
        candidates, 
        eval_data, 
        train_data, # Use train data for few-shot source if needed
        conf['evaluation']
    )

    # 4. Show Results
    print("\n=== Final Leaderboard ===")
    for rank, (prompt, score) in enumerate(scored_results):
        print(f"Rank {rank+1} | Score: {score:.2f} | Prompt: {prompt}")

if __name__ == "__main__":
    main()