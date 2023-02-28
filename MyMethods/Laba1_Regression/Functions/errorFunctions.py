import numpy as np


def euclideError(td, cd, w):
    return (np.dot(td, w) - cd)**2 / td.shape[0]


def manhatanDistanceError(td, cd, w):
    return np.sum((np.dot(td, w) - cd)) / td.shape[0]
