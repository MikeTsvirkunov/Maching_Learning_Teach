import numpy as np


class Layer:
    def __init__(self, 
                 weights: np.ndarray, 
                 dias: float,
                 summator: callable,
                 dsummator: callable,
                 activation_function: callable, 
                 dactivation_function: callable) -> None:
        self.__weights = weights
        self.__dias = dias
        self.summator = summator
        self.dsummator = dsummator
        self.activation_function = activation_function
        self.dactivation_function = dactivation_function
    
    def get_output(self, x: np.array):
        out = np.array([])
        for w in self.__weights:
            s = self.summator(np.append(w.dot(x.T), self.__dias))
            out = np.append(out, self.activation_function(s))
        return out
    
    def get_dias(self):
        return self.__dias
    
    def get_weights(self):
        return self.__weights
    
    def set_dias(self, dias: np.ndarray):
        self.__dias = dias
    
    def set_weights(self, weights: np.ndarray):
        self.__weights = weights

    
    