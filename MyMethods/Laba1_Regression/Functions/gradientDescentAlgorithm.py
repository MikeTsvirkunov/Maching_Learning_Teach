import numpy as np
import numdifftools as nd


def gradientDescentAlgorithm(td, cd, w, lost_function, gradient, merror=0.001, step=0.001, nsteps=10000):
    i = 0
    print(w)

    dfun = nd.Gradient(lambda a: np.sum(lost_function(td, cd, a)))
    ploss = np.sum(lost_function(td, cd, w))
    while i < nsteps:
        pred = np.dot(td, w)
        nloss = np.sum(lost_function(td, cd, w))
        # print(f'grad {i}) ', dfun(w))
        # print(f'{i}) ', w, np.sum(w))
        # print(f'e_funct {i}) ', lost_function(td, cd, w))
        w -= step * dfun(w)
        # x = list()
        # for l in range(td.shape[1]):
        #     x.append(0)
        #     for I in range(td.shape[0]):
        #         x[-1] += td[I][l] * cd[I]
        #         for j in range(td.shape[1]):
        #             x[-1] -= td[I][l] * td[I][j] * w[j]
        # print('\n g:',x)
        # print('\n w1:', w)
        # w -= 2 * np.array(x) * step / td.shape[1]
        # w -= 2 * step * np.sum(np.dot((pred - cd), td)) / td.shape[0] # * (-1**(ploss<=nloss)) 

        # print(nloss, ploss, np.abs(nloss - np.sum(lost_function(td, cd, w))), nloss>ploss)
        # w -= 2 * step * (cd - np.dot(td, w)) / td.shape[0]
        w -= gradient(td, cd, w)
        if np.abs(nloss - np.sum(lost_function(td, cd, w))) < merror:
            return w
        ploss = nloss
        i += 1
    return w