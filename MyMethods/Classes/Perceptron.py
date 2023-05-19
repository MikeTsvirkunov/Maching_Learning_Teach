import numpy as np
from numpy import array
from .Interfaces.IPredictor import IPredictor


class Perceptron(IPredictor):
    def __init__(self, layers_of_weights: iter):#, mass_of_b: iter, activation_functions: iter) -> None:
        super().__init__()
        self.layers = layers_of_weights
    
    def predict(self, x: array) -> array:
        A = x
        for layer in self.layers:
            # print('A, res:\t', A, layer.get_output(A[-1]))
            A = layer.get_output(A)
        return A
