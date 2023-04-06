import numpy as np
from itertools import chain
from .Interfaces.IPredictor import IPredictor


class Regression(IPredictor):
    def __init__(self):
        self.weights = list()
    
    def predict(self, data_for_predict):
        return np.dot(data_for_predict, self.weights)

