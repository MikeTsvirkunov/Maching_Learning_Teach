import numpy as np
from numpy import array
from .Interfaces.IPredictor import IPredictor


class Perceptron(IPredictor):
    def __init__(self, layers_of_weights: iter):#, mass_of_b: iter, activation_functions: iter) -> None:
        super().__init__()
        self.layers = layers_of_weights
    
    def predict(self, x: array) -> array:
        A = x.reshape(1, x.shape[0])
        for layer in self.layers:
            # print(np.dot(A, layer.get_weights().T))
            A = layer.get_activation_function()(np.dot(A, layer.get_weights().T) + layer.get_dias().T)/A.shape[1]
            # print(A.shape, A)
            A = A.reshape(1, A.shape[1])
        return A[0]
