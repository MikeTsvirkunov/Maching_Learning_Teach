import numpy as np


def Lasso(w, a):
    return a * np.sum(np.abs(w))/(w.shape[0])


def Ridge(w, a):
    return a * np.sum(w**2)/w.shape[0]


def RidgeGrad(w, a):
    w2 = list()
    for i in range(w.shape[0]):
        w2.append(2 * w[i] + np.sum(np.delete(w, i)))
    return a * np.array(w2) / w.shape[0]


# def RidgeGradFormula(w, a):
    # w2 = list()
    # for i in range(w.shape[0]):
    #     w2.append(2 * w[i] + np.sum(np.delete(w, i)))
    # return w
