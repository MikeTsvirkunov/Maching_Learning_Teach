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
            s = layer.get_sumation(A)
            # print(s)
            # print()
            A = layer.activation_function(s)
            A = A.reshape(1, A.shape[0])
        return A
