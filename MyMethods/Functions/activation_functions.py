import numpy as np
from Classes.Layer import Layer


def standart_activation(layer: Layer, X: np.array):
    return layer.get_activation_function()(np.dot(X, layer.get_weights().T) + layer.get_diases())


def activation_with_cach(layer: Layer, X: np.array):
    return layer.get_activation_function()(np.dot(X, layer.get_weights().T) + layer.get_diases()), (layer.get_weights(), X, layer.get_diases())


def dactivation_by_dweights(layer: Layer, X: np.array):
    dA = layer.get_dactivation_function()(np.dot(X, layer.get_weights().T) + layer.get_diases())
    return np.dot(dA, X)


def dactivation_by_ddiases(layer: Layer, X: np.array):
    dA = layer.get_dactivation_function()(np.dot(X, layer.get_weights().T) + layer.get_diases())
    return dA
