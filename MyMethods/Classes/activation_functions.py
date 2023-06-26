import pandas as pd
import numpy as np
from scipy.special import expit


class actifation_functions:
    @staticmethod
    def softmax_backword(dA, cache):
        Z = cache
        dZ = dA
        return dZ.T

    @staticmethod
    def relu_backward(dA, cache):

        Z = cache
        dZ = np.ones_like(dA)
        dZ[Z <= 0] = 0
        return dZ

    @staticmethod
    def leaky_relu_derivative(dA, cache):
        Z = cache
        alpha = 0.1
        dZ = np.ones_like(dA)
        dZ[Z < 0] = alpha
        return dZ

    @staticmethod
    def sigmoid_backward_scipy(dA, cache):
        Z = cache
        s = expit(Z)
        dZ = dA * s * (1-s)
        return dZ

    @staticmethod
    def sigmoid_backward(dA, cache):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        return dZ

    @staticmethod
    def lineear_func_back(dA, cache):
        dZ = dA
        return dZ
