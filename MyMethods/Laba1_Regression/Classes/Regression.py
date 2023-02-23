import numpy as np
from itertools import chain

class Regression():
    def __init__(self):
        self.weights = list()
    
    def predict(self, data_for_predict):
        return np.dot(data_for_predict, self.weights)

