import numpy as np


def manhatan_distance(a, b):
    return np.sum(np.abs(a - b))/a.shape[0]


def euclid_distance(a, b):
    return sum(((a-b)**2).T)**0.5/a.shape[0]
