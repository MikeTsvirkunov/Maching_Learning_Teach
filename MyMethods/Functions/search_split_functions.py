from numpy import array, argsort, mean, unique, inf
from collections import Counter


def search_params_that_equiv_to_majority_class(x: array, y: array):
    d = (inf, inf, inf)
    for i in range(x.T.shape[0]):
        for j in x.T[i]:
            y1 = y[x.T[i] > j]
            y2 = y[x.T[i] < j]
            g = y1.shape[0] * gini(y1) / y.shape[0] + y2.shape[0] * gini(y2) / y.shape[0]
            d = min((g, j, i), d, 
                    key=lambda a: a[0])
    return d



def gini(y: array):
    return 1-sum((unique(y, return_counts=True)[1]/y.shape[0])**2)