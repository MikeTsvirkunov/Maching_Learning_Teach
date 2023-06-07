import numpy as np
from ..Classes.Layer import Layer


def standart_activation(layer: Layer, X: np.array):
    return layer.get_activation_function()(np.dot(layer.get_weights(), X) + layer.get_diases())


def activation_with_cach(layer: Layer, X: np.array):
    return layer.get_activation_function()(np.dot(layer.get_weights(), X) + layer.get_diases()), (layer.get_weights(), X, layer.get_diases())
