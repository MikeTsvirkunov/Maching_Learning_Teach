import numpy as np
from .Interfaces.IPredictor import IPredictor


class KNumNeighborsClassifier(IPredictor):
    def __init__(self, k, distance, function_of_priority) -> None:
        self.k = k
        self.distance = distance
        self.function_of_priority = function_of_priority
        self.X_train = None
        self.y_train = None
    
    def predict(self, x: np.array):
        k = list()
        for j in x:
            distances = self.distance(j, self.X_train)
            classes = self.y_train[np.argsort(distances)[:self.k]]
            k.append(self.function_of_priority(list(zip(classes, distances))))
        return np.array(k)

