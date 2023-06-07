import numpy as np
from numpy import array
from .Interfaces.IPredictor import IPredictor


class Perceptron(IPredictor):
    def __init__(self, layers_of_weights: iter):#, mass_of_b: iter, activation_functions: iter) -> None:
        super().__init__()
        self.layers = layers_of_weights
    
    def predict(self, x: array) -> array:
        A = x.reshape(1, x.shape[0])
        # print('A', A.shape, A.tolist())
        for layer in self.layers:
            # print('w', layer.get_weights().shape, layer.get_weights().tolist())
            print(layer.get_weights().dot(A.T).shape)
            # print('d', layer.get_dias().shape, )
            s = layer.get_sumation(A)
            # print('s', s.shape, s.tolist())
            # print()
            A = layer.activation_function(s)
            # print(A.shape, A.tolist())
            A = A.reshape(1, A.shape[1])
        return A
