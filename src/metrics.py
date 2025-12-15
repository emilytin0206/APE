# src/metrics.py
import re
import string
from collections import Counter

TASK_TO_METRIC = {
    'common_concept': 'f1', 'informal_to_formal': 'f1', 
    'orthography_starts_with': 'es', 'taxonomy_animal': 'es', 
    'synonyms': 'contains'
}
DEFAULT_METRIC = 'em'

def normalize_prediction(prediction, lowercase=True):
    prediction = str(prediction)
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip()
    prediction = prediction.split("\n")[0]
    prediction = prediction.split(".")[0]

    if lowercase:
        prediction = prediction.lower()

    # remove punctuation
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(str.maketrans('', '', string.punctuation))
    return prediction

def get_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_prediction(prediction).split()
    ground_truth_tokens = normalize_prediction(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0 or len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def get_em_score(prediction, ground_truth):
    return normalize_prediction(prediction) == normalize_prediction(ground_truth)

def get_exact_set_score(prediction, ground_truth):
    pred = set(normalize_prediction(prediction).split())
    gt = set(normalize_prediction(ground_truth).split())
    return int(pred == gt)

def get_contains_score(prediction, ground_truth):
    pred = normalize_prediction(prediction)
    gt = normalize_prediction(ground_truth)
    if not gt: return 0
    return 1 if re.search(r'\b({0})\b'.format(re.escape(gt)), pred) else 0

def score_function(metric_name, prediction, answers):
    """統一的計分入口"""
    if not isinstance(answers, list):
        answers = [answers]
        
    scores = []
    for answer in answers:
        if metric_name == 'f1':
            scores.append(get_f1_score(prediction, answer))
        elif metric_name == 'es':
            scores.append(get_exact_set_score(prediction, answer))
        elif metric_name == 'contains':
            scores.append(get_contains_score(prediction, answer))
        else: # em (Exact Match)
            scores.append(1 if get_em_score(prediction, answer) else 0)
    
    return max(scores) if scores else 0