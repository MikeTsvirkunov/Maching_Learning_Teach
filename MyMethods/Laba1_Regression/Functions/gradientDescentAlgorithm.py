import numpy as np


def gradientDescentAlgorithm(td, cd, w, ferror, merror=0.001, step=0.001, nsteps=100):
    i = 0
    while i < nsteps:
        pred = np.matmul(td, w)
        loss = np.sum((cd - pred)**2) / td.shape[0]
        w -= step * np.sum(np.matmul((pred - cd), td)) * 2 / td.shape[0]
        if ferror(td, cd, w, loss) < merror:
            return w 
        i += 1
    return w