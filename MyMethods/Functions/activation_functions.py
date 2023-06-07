import numpy as np
from Classes.Layer import Layer


def standart_activation(layer: Layer, X: np.array):
    return layer.get_activation_function()(np.dot(X, layer.get_weights().T) + layer.get_diases())


def activation_with_cach(layer: Layer, X: np.array):
    return layer.get_activation_function()(np.dot(X, layer.get_weights().T) + layer.get_diases()), (layer.get_weights(), X, layer.get_diases())


def dactivation_dw(layer: Layer, X: np.array, cach: tuple, output):
    linear_cache, activation_cache = output
    
    layer.get_dactivation_function()()
    return layer.get_dactivation_function()(np.dot(X, layer.get_weights().T) + layer.get_diases())
