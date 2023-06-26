import numpy as np


class Layer:
    def __init__(self, 
                 weights: np.ndarray, 
                 dias: np.array,
                 activation_function: callable, 
                 dactivation_function: callable) -> None:
        self.__weights = weights
        self.__dias = dias
        self.activation_function = activation_function
        self.dactivation_function = dactivation_function

    def get_activation_function(self):
        return self.activation_function
    
    def get_dactivation_function(self):
        return self.dactivation_function

    def get_dias(self):
        return self.__dias
    
    def get_weights(self):
        return self.__weights
    
    def set_dias(self, dias: np.ndarray):
        self.__dias = dias
    
    def set_weights(self, weights: np.ndarray):
        self.__weights = weights

    
    