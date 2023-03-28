import numpy as np
from .Interfaces.IPredictor import IPredictor


class CartTreeClassifier(IPredictor):
    def __init__(self, function_of_priority) -> None:
        self.function_of_priority = function_of_priority
        self.X_train = None
        self.y_train = None
        self.spreading_functions = None
    
    def predict(self, x) -> np.array:
        pass

