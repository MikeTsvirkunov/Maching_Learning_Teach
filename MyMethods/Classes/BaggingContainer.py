import numpy as np
from numpy import array
from .Interfaces.IPredictor import IPredictor
from .Interfaces.IBag import IBag


class BaggingContainer(IPredictor, IBag):
    def __init__(self, 
                 predictors: iter, 
                 out_function: callable,) -> None:
        self.__predictors = predictors
        self.__out_function = out_function
    
    def get_bag(self) -> any:
        return self.__predictors

    def predict(self, x: array) -> array:
        d = list()
        for predictor in self.__predictors:
            d.append(predictor.predict(x))
        return self.__out_function(np.array(d))
