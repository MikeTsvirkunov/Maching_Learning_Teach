import numpy as np


def euclideError(td, cd, w, loss):
    return np.abs(loss - np.sum((np.matmul(td, w) - cd)**2) / td.shape[0])


def manhatanDistanceError(td, cd, w, loss):
    return np.abs(loss - np.sum((np.matmul(td, w) - cd)) / td.shape[0])
