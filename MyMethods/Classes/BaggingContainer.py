import numpy as np
from numpy import array
from .Interfaces.IPredictor import IPredictor
from .Interfaces.IContainer import IContainer
from .Node import Node


class BaggingContainer(IPredictor, IContainer):
    def __init__(self, predictors: iter, out_function: callable, split_function: callable) -> None:
        self.__predictors = predictors
        self.__out_function = out_function
        self.__split_function = split_function

    def predict(self, x: array) -> array:
        return self.__split_function(*map(lambda p, d: p.predict(d), zip(self.__predictors, self.__split_function())))
