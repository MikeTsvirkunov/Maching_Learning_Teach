import numpy as np


class Layer:
    def __init__(self, 
                 weights: np.ndarray, 
                 diases: np.array,
                 activation_function: callable,
                 dactivation_function: callable) -> None:
        self.__weights = weights
        self.__diases = diases
        self.activation_function = activation_function
        self.dactivation_function = dactivation_function

    def get_dias(self):
        return np.array(self.__diases, copy=True)
    
    def get_weights(self):
        return np.array(self.__weights, copy=True)
    
    def set_dias(self, diases: np.ndarray):
        self.__diases = np.array(diases, copy=True)
    
    def set_weights(self, weights: np.ndarray):
        self.__weights = np.array(weights, copy=True)

    def get_activation_function(self):
        return self.activation_function
    
    def get_dactivation_function(self):
        return self.dactivation_function
