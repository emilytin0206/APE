# ape/utility.py
import re
import string
from collections import Counter

def normalize_prediction(prediction, lowercase=True):
    """
    Standard APE normalization logic.
    Strips punctuation, handles newlines, and lowercases.
    """
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

def get_em_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(ground_truth, lowercase=True)
    return int(prediction_normalized == ground_truth_normalized)

def get_multi_answer_em(prediction, answers):
    """Checks if prediction matches ANY valid answer in the list."""
    for answer in answers:
        if get_em_score(prediction, answer) == 1:
            return 1
    return 0

# (保留其他 metric 函數以維持相容性，如 f1, contains 等)