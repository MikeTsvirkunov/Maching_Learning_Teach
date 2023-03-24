import numpy as np

def normal_spread(v, t: np.array):
    return 1 / np.sqrt(np.std(t) * 2 * np.pi) * np.exp(- 0.5 * ((v - np.mean(t)) / np.std(t))**2) 