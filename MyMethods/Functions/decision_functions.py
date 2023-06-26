import numpy as np

def mean_value(y: np.array):
    return sum(y) // (y.shape[0]//2+1)
