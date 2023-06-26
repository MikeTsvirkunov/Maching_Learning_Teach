import numpy as np
# import numdifftools as nd

# dfun = nd.Gradient(lambda a: np.sum(lost_function(td, cd, a)))


def gradientDescentAlgorithm(td, cd, w, lost_function, gradient, merror=0.001, step=0.001, nsteps=100):
    i = 0
    while i < nsteps:
        loss = lost_function(td, cd, w)
        w -= gradient(td, cd, w) * step
        if np.abs(loss - lost_function(td, cd, w)) < merror:
            return w
        i += 1
    return w


def tailGradientDescentAlgorithm(td, cd, w, lost_function, gradient, merror=0.001, step=0.001, nsteps=100):
    if np.abs(lost_function(td, cd, w) - lost_function(td, cd, w-gradient(td, cd, w) * step)) < merror or not nsteps:
        return w
    return tailGradientDescentAlgorithm(td, cd, 
                                        w-gradient(td, cd, w) * step, 
                                        lost_function, gradient, 
                                        merror=0.001, step=0.001, nsteps=nsteps-1)
