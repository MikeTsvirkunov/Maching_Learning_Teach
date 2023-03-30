import numpy as np
from numpy import array
from .Interfaces.IPredictor import IPredictor
from .Node import Node


class CartTreeClassifier(IPredictor):
    def __init__(self) -> None:
        self.X_train = None
        self.y_train = None
        self.spreading_functions = None
        self.tree = None

    def predict(self, x: array) -> array:
        t = self.tree.next(x[0])
        print(x.shape[0])
        for i in range(1, x.shape[0]):
            t = t.next(x[i])
            print('->', i, t)
        return t

