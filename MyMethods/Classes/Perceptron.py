import numpy as np

class Perceptron():
    def __init__(self, layers, activation_function):
        self.layers = layers
        self.activation_function = activation_function

    def predict(self, X: np.array):
        output = np.array(X, copy=True)
        for layer in self.layers:
            output = self.activation_function(layer, output)
        return output
