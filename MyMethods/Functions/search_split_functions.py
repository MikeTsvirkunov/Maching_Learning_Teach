from numpy import array, argsort, mean, unique, inf
from Functions.metrics import gini
from math import log

def ie(y):
    cs = [y[y==i] for i in unique(y)]
    return sum(-log(len(i)/y.shape[0])*len(i)/y.shape[0] for i in cs)



def search_params_that_equiv_to_majority_class(x: array, y: array, err=0):
    d = (inf, inf, inf)
    for i in range(x.T.shape[0]):
        for j in x.T[i]:
            y1 = y[x.T[i] >= j]
            y2 = y[x.T[i] < j]
            
            g = y1.shape[0] * gini(y1) / y.shape[0] + y2.shape[0] * gini(y2) / y.shape[0]
            d = min((g, j, i), d, 
                    key=lambda a: a[0])
            if d[0]<=err:
                return d
    return d


def search_params_in_no_district(x: array, y: array, err=0):
    d = (inf, inf, inf)
    for i in range(x.T.shape[0]):
        for j in x.T[i]:
            y1 = y[x.T[i] >= j]
            y2 = y[x.T[i] < j]

            g = (y1.shape[0] * ie(y) + y1.shape[0] * ie(y)) / y.shape[0]
            d = min((g, j, i), d,
                    key=lambda a: a[0])
            if d[0] <= err:
                return d
    return d


def search_params_dispertinised(x: array, y: array, err=0):
    d = (inf, inf, inf)
    k = dis(y)
    for i in range(x.T.shape[0]):
        for j in x.T[i]:
            y1 = y[x.T[i] >= j]
            y2 = y[x.T[i] < j]

            g = k - (dis(y1) + dis(y2))
            d = min((g, j, i), d,
                    key=lambda a: a[0])
            if d[0] <= err:
                return d
    return d


def dis(y):
    r = 0
    for i in y:
        for j in y:
            r += 0.5 * (i - j)**2
    return r / (y.shape[0]**2)
