import numpy as np
from numpy import array
from .Interfaces.IPredictor import IPredictor
from .Node import Node


class CartTreeClassifier(IPredictor):
    def __init__(self) -> None:
        self.tree = None

    def predict(self, x: array) -> array:
        pred = list()
        for i in x:
            t = self.tree
            while type(t) is Node:
                t = t.next(i)
            pred.append(t)
        return np.array(pred)

