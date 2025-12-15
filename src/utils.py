# src/utils.py
import random

def subsample_data(data, n):
    """
    data: tuple (inputs, outputs)
    n: number of samples
    """
    inputs, outputs = data
    if n >= len(inputs):
        return inputs, outputs
    
    indices = random.sample(range(len(inputs)), n)
    return ([inputs[i] for i in indices], [outputs[i] for i in indices])