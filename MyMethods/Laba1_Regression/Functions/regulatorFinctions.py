import numpy as np


def Lasso(w, a):
    return a * np.sum(np.abs(w))/(w.shape[0])


def Ridge(w, a):
    return a * np.sum(w**2)/w.shape[0]
